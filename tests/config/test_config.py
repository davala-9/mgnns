import yaml
import pytest
from pathlib import Path

from src.config.config import ExperimentConfig, EncoderType, AggregationType

# Helper function that writes a YAML config file for certain given parameters
def write_config(tmp_path, **overrides):
    data_dir = tmp_path / "data"
    exp_dir = tmp_path / "experiments"

    data_dir.mkdir()
    exp_dir.mkdir()

    # Default values
    config = {
        "data_dir": str(data_dir),
        "exp_dir": str(exp_dir),
        "encoding_scheme": "canonical",
        "agg_function_1": "max",
        "agg_function_2": "sum",
        "derivation_threshold": 0.5,
        "clamping": 0.1,
        "use_dummies": True,
        "non_negative_weights": False,
    }

    # Overwrite default values with any extra given arguments
    config.update(overrides)

    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f)

    return config_file


def test_valid_config_loads(tmp_path):
    config_file = write_config(tmp_path)

    cfg = ExperimentConfig(str(config_file))

    assert cfg.encoding_scheme == EncoderType.CANONICAL
    assert cfg.agg_function_1 == AggregationType.MAX
    assert cfg.agg_function_2 == AggregationType.SUM
    assert cfg.derivation_threshold == 0.5
    assert cfg.clamping == 0.1
    assert cfg.use_dummies is True
    assert cfg.non_negative_weights is False


def test_invalid_data_dir_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        data_dir=str(tmp_path / "missing_directory"),
    )

    with pytest.raises(ValueError, match="data path"):
        ExperimentConfig(str(config_file))


def test_invalid_exp_dir_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        exp_dir=str(tmp_path / "missing_directory"),
    )

    with pytest.raises(ValueError, match="experiment path"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("field,match", [
    ("data_dir", "data path"),
    ("exp_dir", "experiment path"),
])
def test_dir_field_pointing_at_file_raises(tmp_path, field, match):
    a_file = tmp_path / "not_a_directory"
    a_file.write_text("")
    config_file = write_config(tmp_path, **{field: str(a_file)})

    with pytest.raises(ValueError, match=match):
        ExperimentConfig(str(config_file))


def test_invalid_encoder_type_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        encoding_scheme="invalid_encoder",
    )

    with pytest.raises(ValueError, match="encoder type not valid"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("field", ["agg_function_1", "agg_function_2"])
def test_invalid_aggregation_type_raises(tmp_path, field):
    config_file = write_config(tmp_path, **{field: "average"})

    with pytest.raises(ValueError, match="aggregation function not valid"):
        ExperimentConfig(str(config_file))


def test_threshold_out_of_range_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        derivation_threshold=1.5,
    )

    with pytest.raises(ValueError, match="between 0 and 1"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_boundary_values_accepted(tmp_path, threshold):
    config_file = write_config(
        tmp_path,
        derivation_threshold=threshold,
    )

    cfg = ExperimentConfig(str(config_file))

    assert cfg.derivation_threshold == threshold


def test_clamping_zero_accepted(tmp_path):
    config_file = write_config(
        tmp_path,
        clamping=0,
    )

    cfg = ExperimentConfig(str(config_file))

    assert cfg.clamping == 0.0


def test_threshold_not_float_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        derivation_threshold="not_a_number",
    )

    with pytest.raises(ValueError, match="threshold value must be a float"):
        ExperimentConfig(str(config_file))


def test_negative_clamping_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        clamping=-0.5,
    )

    with pytest.raises(ValueError, match="clamping value must be non-negative"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("missing_key", [
    "data_dir",
    "exp_dir",
    "encoding_scheme",
    "agg_function_1",
    "agg_function_2",
    "derivation_threshold",
    "clamping",
    "use_dummies",
    "non_negative_weights",
])
def test_missing_key_raises(tmp_path, missing_key):
    config_file = write_config(tmp_path)
    data = yaml.safe_load(config_file.read_text())
    del data[missing_key]
    config_file.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match=f"missing required config key: '{missing_key}'"):
        ExperimentConfig(str(config_file))


def test_null_threshold_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        derivation_threshold=None,
    )

    with pytest.raises(ValueError, match="threshold value must be a float"):
        ExperimentConfig(str(config_file))


def test_null_clamping_raises(tmp_path):
    config_file = write_config(
        tmp_path,
        clamping=None,
    )

    with pytest.raises(ValueError, match="clamping value must be a float"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("field", ["use_dummies", "non_negative_weights"])
def test_non_bool_flag_raises(tmp_path, field):
    config_file = write_config(tmp_path, **{field: "yes"})

    with pytest.raises(ValueError, match=f"{field} value must be true or false"):
        ExperimentConfig(str(config_file))


@pytest.mark.parametrize("field", ["use_dummies", "non_negative_weights"])
def test_int_flag_raises(tmp_path, field):
    # bool is a subclass of int in Python, so 1/0 could slip past a naive check;
    # YAML also parses 1/0 as ints, not bools, so this must be rejected.
    config_file = write_config(tmp_path, **{field: 1})

    with pytest.raises(ValueError, match=f"{field} value must be true or false"):
        ExperimentConfig(str(config_file))


def test_invalid_encoder_type_chains_original_exception(tmp_path):
    config_file = write_config(
        tmp_path,
        encoding_scheme="invalid_encoder",
    )

    with pytest.raises(ValueError) as exc_info:
        ExperimentConfig(str(config_file))

    assert isinstance(exc_info.value.__cause__, ValueError)