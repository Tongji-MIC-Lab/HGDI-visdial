import argparse
import json
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders import Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint


parser = argparse.ArgumentParser(
    "Evaluate and/or generate EvalAI submission file."
)
parser.add_argument(
    "--config-yml",
    default="configs/hgdi.yml",
)
parser.add_argument(
    "--split",
    default="val",
    choices=["val", "test"],
)
parser.add_argument(
    "--val-json",
    default="./VisDial_1.0/visdial_1.0_val.json",
)
parser.add_argument(
    "--val-structure-json",
    default="./visdial_1.0_val_coref_structure.json"
)
parser.add_argument(
    "--val-dense-json",
    default="./visdial_1.0_val_dense_annotations.json",
)
parser.add_argument(
    "--test-json",
    default="./visdial_1.0_test.json",
)
parser.add_argument(
    "--test-structure-json",
    default="./visdial_1.0_test_coref_structure.json"
)
parser.add_argument_group("Evaluation related arguments")
parser.add_argument(
    "--load-pthpath",
    default="",
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
    action="store_true",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
)
parser.add_argument(
    "--save-scores",
    default=False
)
parser.add_argument(
    "--save-ranks",
    default=True
)
parser.add_argument(
    "--seed",
    type=int,
    default=0
)
parser.add_argument_group("Submission related arguments")
parser.add_argument(
    "--save-ranks-path",
    default="logs/ranks.json",
)


# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))

# =============================================================================
#   SETUP DATASET, DATALOADER, MODEL
# =============================================================================

val_dataset = VisDialDataset(
    config=config["dataset"],
    dialogs_jsonpath=args.val_json,
    coref_dependencies_jsonpath=args.val_structure_json,
    dense_annotations_jsonpath=args.val_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    return_options=True,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
)

if args.split == "val":
    val_dataset = VisDialDataset(
        config=config["dataset"],
        dialogs_jsonpath=args.val_json,
        coref_dependencies_jsonpath=args.val_structure_json,
        dense_annotations_jsonpath=args.val_dense_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] == "disc"
        else True,
    )
else:
    val_dataset = VisDialDataset(
        config=config["dataset"],
        dialogs_jsonpath=args.test_json,
        coref_dependencies_jsonpath=args.test_structure_json,
        overfit=args.overfit,
        in_memory=args.in_memory,
        return_options=True,
        add_boundary_toks=False
        if config["model"]["decoder"] == "disc"
        else True,
    )
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] == "disc"
    else 5,
    num_workers=args.cpu_workers,
)

encoder = Encoder(config["model"], val_dataset.vocabulary)
decoder = Decoder(config["model"], val_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

decoder.word_embed = encoder.word_embed

model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

model_state_dict, _ = load_checkpoint(args.load_pthpath)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
print("Loaded model from {}".format(args.load_pthpath))

sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# =============================================================================
#   EVALUATION LOOP
# =============================================================================

model.eval()
ranks_json = []
scores = None

for _, batch in enumerate(tqdm(val_dataloader)):
    for key in batch:
        if not isinstance(batch[key], list):
            batch[key] = batch[key].cuda()
    with torch.no_grad():
        _, output, _, _, _ = model(batch)

    if args.save_scores:
        if scores is None:
            scores = output
        else:
            scores = torch.cat((scores, output), dim=0)
    else:
        ranks = scores_to_ranks(output)
        for i in range(len(batch["img_ids"])):
            if args.split == "test":
                ranks_json.append(
                    {
                        "image_id": batch["img_ids"][i].item(),
                        "round_id": int(batch["num_rounds"][i].item()),
                        "ranks": [
                            rank.item()
                            for rank in ranks[i][batch["num_rounds"][i] - 1]
                        ],
                    }
                )
            else:
                for j in range(batch["num_rounds"][i]):
                    ranks_json.append(
                        {
                            "image_id": batch["img_ids"][i].item(),
                            "round_id": int(j + 1),
                            "ranks": [rank.item() for rank in ranks[i][j]],
                        }
                    )

        if args.split == "val":
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[
                         torch.arange(output.size(0)), batch["round_id"] - 1, :
                         ]
                ndcg.observe(output, batch["gt_relevance"])

if args.save_scores:
    print(scores.size())
    torch.save(scores, "checkpoints/scores/{}/{}.pt".format(args.split, args.load_pthpath.split('/')[-1].split('.')[0]))
else:
    if args.split == "val":
        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            print(f"{metric_name}: {metric_value}")

    print("Writing ranks to {}".format(args.save_ranks_path))
    os.makedirs(os.path.dirname(args.save_ranks_path), exist_ok=True)
    if args.save_ranks:
        json.dump(ranks_json, open(args.save_ranks_path, "w"))


