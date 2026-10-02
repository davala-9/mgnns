"""
Where runs write their output. Training creates a model folder in the configuration's models_dir, holding the model,
its encoders, its checkpoints and a copy of the configuration. Every other script is an experiment on a trained model:
it creates an experiment folder in the exp_dir of that model's configuration, named after the dataset, the experiment
and the start time, with a run.yaml recording how it was run.
"""
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from src.config.config import EncoderType, ModelConfig
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.utils.utils import check

MODEL_CONFIG_FILE = "config.yaml"
RUN_FILE = "run.yaml"

def _timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# Creates the folder for a model trained with cfg (read from config_path), e.g. models/WN18RRv1_maxmax_[time].
def create_model_folder(cfg: ModelConfig, config_path) -> Path:
    folder = cfg.models_dir / f"{cfg.data_dir.name}_{cfg.agg_function_1.value}{cfg.agg_function_2.value}_{_timestamp()}"
    folder.mkdir(parents=True)
    shutil.copy(config_path, folder / MODEL_CONFIG_FILE)
    return folder

def load_model_config(model_folder) -> ModelConfig:
    return ModelConfig(check(Path(model_folder) / MODEL_CONFIG_FILE, "Model configuration"))

def load_model(model_folder, device):
    return torch.load(check(Path(model_folder) / "model.pt", "Model"), weights_only=False,
                      map_location=device).to(device)

def load_encoder(model_folder, encoding_scheme: EncoderType):
    print("Loading encoder from file...")
    folder = Path(model_folder)
    internal_encoder = CanonicalEncoderDecoder(check(folder / 'internal_encoder.tsv', "Internal encoding"))
    if encoding_scheme == EncoderType.ICLR22:
        external_encoder = ICLREncoderDecoder(check(folder / 'external_encoder.tsv', "External encoding"))
    else:
        external_encoder = IdentityEncoderDecoder(check(folder / 'external_encoder.tsv', "External encoding"))
    return external_encoder, internal_encoder

# The commit the code was run from, and whether the working tree had uncommitted changes (None if not a git checkout).
def _git_state():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    return commit, (None if commit is None else bool(git("status", "--porcelain")))

# Creates the folder for one run of `experiment` on the model in model_folder, e.g.
# experiments/WN18RRv1_evaluate_[time], and returns the model's configuration along with it.
def create_experiment_folder(experiment: str, model_folder) -> tuple[ModelConfig, Path]:
    cfg = load_model_config(model_folder)
    folder = cfg.exp_dir / f"{cfg.data_dir.name}_{experiment}_{_timestamp()}"
    folder.mkdir(parents=True)
    commit, uncommitted_changes = _git_state()
    with open(folder / RUN_FILE, 'w') as f:
        yaml.safe_dump({"experiment": experiment,
                        "model": str(model_folder),
                        "command": shlex.join(sys.orig_argv),
                        "git_commit": commit,
                        "uncommitted_changes": uncommitted_changes}, f, sort_keys=False, width=float("inf"))
    return cfg, folder

# Adds a value chosen during the run (e.g. a threshold computed from the data) to the experiment's run.yaml.
def record(experiment_folder: Path, key: str, value):
    with open(experiment_folder / RUN_FILE) as f:
        run = yaml.safe_load(f)
    run[key] = value
    with open(experiment_folder / RUN_FILE, 'w') as f:
        yaml.safe_dump(run, f, sort_keys=False, width=float("inf"))
