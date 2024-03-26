from pathlib import Path
from subprocess import PIPE, Popen
import warnings

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):

    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dirpath,
        step_size=1,
        last_epoch=0,
        **kwargs,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError(
                "{} is not an Optimizer".format(type(optimizer).__name__)
            )

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.init_directory(**kwargs)

    def init_directory(self, config={}):
        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")
        commit_sha_filepath = self.ckpt_dirpath / f".commit-{commit_sha}"
        commit_sha_filepath.touch()
        yaml.dump(
            config,
            open(str(self.ckpt_dirpath / "config.yml"), "w"),
            default_flow_style=False,
        )

    def step(self, epoch=None):

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if not self.last_epoch % self.step_size:
            torch.save(
                {
                    "model": self._model_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
            )

    def _model_state_dict(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath):

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))

    if len(checkpoint_commit_sha) == 0:
        warnings.warn(
            "Commit SHA was not recorded while saving checkpoints."
        )
    else:
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )

    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]
