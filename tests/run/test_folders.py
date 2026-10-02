import re
import sys

import yaml

from src.config.config import ModelConfig
from src.run.folders import MODEL_CONFIG_FILE, RUN_FILE, create_experiment_folder, create_model_folder, record


def write_config(tmp_path):
    for folder in ["WN18RRv1", "models", "experiments"]:
        (tmp_path / folder).mkdir()
    config = {"data_dir": str(tmp_path / "WN18RRv1"), "models_dir": str(tmp_path / "models"),
              "exp_dir": str(tmp_path / "experiments"), "encoding_scheme": "iclr22", "agg_function_1": "max",
              "agg_function_2": "sum", "clamping": 0, "use_dummies": False, "non_negative_weights": True}
    config_file = tmp_path / "my_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    return config_file


def test_model_folder_is_named_after_dataset_and_aggregations_and_holds_the_config(tmp_path):
    config_file = write_config(tmp_path)
    folder = create_model_folder(ModelConfig(str(config_file)), config_file)
    assert folder.parent == tmp_path / "models"
    assert re.fullmatch(r"WN18RRv1_maxsum_\d{8}_\d{6}", folder.name)
    assert (folder / MODEL_CONFIG_FILE).read_text() == config_file.read_text()


def test_experiment_folder_uses_the_model_config_and_records_the_run(tmp_path, monkeypatch):
    config_file = write_config(tmp_path)
    model_folder = create_model_folder(ModelConfig(str(config_file)), config_file)
    monkeypatch.setattr(sys, "orig_argv", ["python", "-m", "src.run.evaluate", "--model", str(model_folder)])
    cfg, folder = create_experiment_folder("evaluate", model_folder)
    assert cfg.data_dir == tmp_path / "WN18RRv1"
    assert folder.parent == tmp_path / "experiments"
    assert re.fullmatch(r"WN18RRv1_evaluate_\d{8}_\d{6}", folder.name)
    run = yaml.safe_load((folder / RUN_FILE).read_text())
    assert run["experiment"] == "evaluate"
    assert run["model"] == str(model_folder)
    assert run["command"] == f"python -m src.run.evaluate --model {model_folder}"
    assert {"git_commit", "uncommitted_changes"} <= run.keys()


def test_record_adds_a_value_to_run_file(tmp_path):
    config_file = write_config(tmp_path)
    model_folder = create_model_folder(ModelConfig(str(config_file)), config_file)
    _, folder = create_experiment_folder("evaluate", model_folder)
    record(folder, "extraction_threshold", 0.02)
    run = yaml.safe_load((folder / RUN_FILE).read_text())
    assert run["extraction_threshold"] == 0.02 and run["experiment"] == "evaluate"
