import os
import argparse
import itertools
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import datetime
import random

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import CheckpointManager, load_checkpoint
from visdialch.utils.logging import Logger
from visdialch.utils.scheduler import get_optim, adjust_lr

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/hgdi.yml",
)
parser.add_argument(
    "--train-json",
    default="./visdial_1.0_train.json",
)
parser.add_argument(
    "--val-json",
    default="/./visdial_1.0_val.json",
)
parser.add_argument(
    "--train-structure-json",
    default="./visdial_1.0_train_coref_structure.json"
)
parser.add_argument(
    "--val-structure-json",
    default="./visdial_1.0_val_coref_structure.json"
)
parser.add_argument(
    "--train-neural-dense-json",
    default="./visdial_1.0_train_dense_labels.json"
)
parser.add_argument(
    "--val-dense-json",
    default="./visdial_1.0_val_dense_annotations.json",
)

parser.add_argument_group(
    "Arguments independent of experiment reproducibility"
)
parser.add_argument(
    "--gpu-ids",
    nargs="+",
    type=int,
    default=[0, 1],
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=8,
)
parser.add_argument(
    "--overfit",
    default=False,
)
parser.add_argument(
    "--validate",
    default=True,
)
parser.add_argument(
    "--in-memory",
    default=False,
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
)
parser.add_argument(
    "--load-pthpath",
    default="",
)

# ==========================================================================
#   RANDOM SEED
# ==========================================================================
seed = random.randint(0, 99999999)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

# ==========================================================================
#   INPUT ARGUMENTS AND CONFIG
# ==========================================================================
args = parser.parse_args()
config = yaml.safe_load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# ==========================================================================
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# ==========================================================================
train_dataset = VisDialDataset(
    config=config["dataset"],
    dialogs_jsonpath=args.train_json,
    coref_dependencies_jsonpath=args.train_structure_json,
    answer_plausibility_jsonpath=args.train_neural_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True if config["model"]["decoder"] == "disc" else False,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
    sample_flag = False,
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
    drop_last=True,
)
val_dataset = VisDialDataset(
    config=config["dataset"],
    dialogs_jsonpath=args.val_json,
    coref_dependencies_jsonpath=args.val_structure_json,
    dense_annotations_jsonpath=args.val_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
    sample_flag = False,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] == "disc"
    else 5,
    num_workers=args.cpu_workers,
    drop_last=True,
)

encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

if config["dataset"]["glove_npy"] != '':
    encoder.word_embed.weight.data = torch.from_numpy(np.load(config["dataset"]["glove_npy"]))
    print("Loaded glove vectors from {}".format(config["dataset"]["glove_npy"]))

decoder.word_embed = encoder.word_embed

model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)


if config["model"]["decoder"] == "disc":
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.MSELoss()
elif config["model"]["decoder"] == "gen":
    criterion2 = nn.CrossEntropyLoss(
        ignore_index=train_dataset.vocabulary.PAD_INDEX
    )
    criterion1 = nn.BCEWithLogitsLoss()
    criterion3 = nn.MSELoss()
else:
    raise NotImplementedError

if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // config["solver"][
        "batch_size"
    ] + 1
else:
    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1

# ==========================================================================
#   SETUP BEFORE TRAINING LOOP
# ==========================================================================
start_time = datetime.datetime.strftime(datetime.datetime.utcnow(), '%d-%b-%Y-%H:%M:%S')
if args.save_dirpath == 'checkpoints/':
    args.save_dirpath += '%s' % start_time

os.makedirs(args.save_dirpath, exist_ok=True)
logger = Logger(os.path.join(args.save_dirpath, 'log.txt'))
logger.write("{}".format(seed))

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

if args.load_pthpath == "":
    start_epoch = 1
    optim = get_optim(config, model, len(train_dataset))

else:
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    optim = get_optim(config, model, len(train_dataset))
    optim._step = iterations * (start_epoch - 1)
    optim.optimizer.load_state_dict(optimizer_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))

checkpoint_manager = CheckpointManager(
    model, optim.optimizer, args.save_dirpath, last_epoch=start_epoch - 1, config=config
)

# =============================================================================
#   TRAINING LOOP
# =============================================================================
running_loss = 0.0
train_begin = datetime.datetime.utcnow()
for epoch in range(start_epoch, config["solver"]["num_epochs"] + 1):
    # -------------------------------------------------------------------------
    #   ADJUST LEARNING RATE
    # -------------------------------------------------------------------------
    if epoch in config["solver"]["lr_decay_list"]:
        adjust_lr(optim, config["solver"]["lr_decay_rate"])

    # -------------------------------------------------------------------------
    #   ON EPOCH START
    # -------------------------------------------------------------------------
    combined_dataloader = itertools.chain(train_dataloader)

    print(f"\nTraining for epoch {epoch}:")
    for i, batch in enumerate(combined_dataloader):
        for key in batch:
            if not isinstance(batch[key], list):
                batch[key] = batch[key].cuda()

        optim.zero_grad()
        encoder_output, output, structures, sim_paths, sim_label = model(batch)

        output1 = output

        target = (
            batch["ans_ind"]
            if config["model"]["decoder"] == "disc"
            else batch["ans_out"]
        )

        if epoch < 5:
            loss1 = criterion2(output1.view(-1, output.size(-1)), target.view(-1))
            output.view(-1, output.size(-1))
        else:
            loss1 = criterion1(output1, batch["teacher_scores"])
        loss2 = criterion3(structures, batch["structures"])
        path_sim_loss = ((sim_paths - sim_label) ** 2).sum() /config["solver"]["batch_size"]

        batch_loss = loss1 + loss2 + path_sim_loss

        batch_loss.backward()
        optim.step()

        # --------------------------------------------------------------------
        # update running loss and decay learning rates
        # --------------------------------------------------------------------
        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
        else:
            running_loss = batch_loss.item()

        torch.cuda.empty_cache()
        if i % 100 == 0:
            # print current time, running average, learning rate, iteration, epoch
            logger.write("[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
                datetime.datetime.utcnow() - train_begin, epoch,
                (epoch - 1) * iterations + i, running_loss,
                optim.optimizer.param_groups[0]['lr']))

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    checkpoint_manager.step()

    if args.validate:
        model.eval()
        logger.write("\nValidation after epoch {}:".format(epoch))
        total_hist_usage = 0
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                if not isinstance(batch[key], list):
                    batch[key] = batch[key].cuda()
            with torch.no_grad():
                encoder_output, output, structures, sim_paths, sim_label = model(batch)
                total_hist_usage += torch.sum(structures)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[
                         torch.arange(output.size(0)), batch["round_id"] - 1, :
                         ]
                ndcg.observe(output, batch["gt_relevance"])

        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            logger.write("{}: {:4f}".format(metric_name, metric_value))
        model.train()
        torch.cuda.empty_cache()