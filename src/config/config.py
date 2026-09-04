import yaml
from enum import Enum
from pathlib import Path

class EncoderType(Enum):
    CANONICAL = "canonical"
    ICLR22 = "iclr22"

class AggregationType(Enum):
    MAX = "max"
    SUM = "sum"

def _require(data: dict, key: str):
    if key not in data:
        raise ValueError(f"missing required config key: {key!r}")
    return data[key]


def _require_float(data: dict, key: str) -> float:
    value = _require(data, key)
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} value must be a float, got {value!r}") from e


def _require_bool(data: dict, key: str) -> bool:
    value = _require(data, key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} value must be true or false, got {value!r}")
    return value


class ExperimentConfig:

    data_dir: Path  # Folder with the specific dataset with all training, validation, and test data.
    exp_dir: Path  # Folder where all experiment folders are stored (we will create a new experiment folder in here)
    encoding_scheme: EncoderType  # Encoding/decoding scheme (canonical or iclr22)
    agg_function_1: AggregationType  # Aggregation functions for layer 1
    agg_function_2: AggregationType  # Aggregation functions for layer 2
    derivation_threshold: float # Model threshold for derivation; must be between 0 and 1
    use_dummies: bool  # Use dummy nodes during training (this is a training optimisation that sometimes helps)
    clamping: float  # Clamp weights whose absolute value is smaller than this to 0. [CURRENTLY UNSUPPORTED]
    non_negative_weights: bool  # Use only non-negative weights in the model's matrices.

    def __init__(self, config_path: str):

        with open(config_path) as f:
            data = yaml.safe_load(f)

        data_path = Path(_require(data, "data_dir"))
        if not data_path.is_dir():
            raise ValueError(f"data path is not an existing folder: {data_path}")
        self.data_dir = data_path

        exp_path = Path(_require(data, "exp_dir"))
        if not exp_path.is_dir():
            raise ValueError(f"experiment path is not an existing folder: {exp_path}")
        self.exp_dir = exp_path

        encoding_scheme = _require(data, "encoding_scheme")
        try:
            self.encoding_scheme = EncoderType(encoding_scheme)
        except ValueError as e:
            valid = " or ".join(f'"{t.value}"' for t in EncoderType)
            raise ValueError(f"encoder type not valid: please choose {valid}") from e

        agg_function_1 = _require(data, "agg_function_1")
        try:
            self.agg_function_1 = AggregationType(agg_function_1)
        except ValueError as e:
            valid = " or ".join(f'"{t.value}"' for t in AggregationType)
            raise ValueError(f"aggregation function not valid: please choose {valid}") from e

        agg_function_2 = _require(data, "agg_function_2")
        try:
            self.agg_function_2 = AggregationType(agg_function_2)
        except ValueError as e:
            valid = " or ".join(f'"{t.value}"' for t in AggregationType)
            raise ValueError(f"aggregation function not valid: please choose {valid}") from e

        self.derivation_threshold = _require_float(data, "derivation_threshold")
        if not 0 <= self.derivation_threshold <= 1:
            raise ValueError(f"threshold value must be between 0 and 1, got {self.derivation_threshold!r}")

        self.clamping = _require_float(data, "clamping")
        if self.clamping < 0:
            raise ValueError(f"clamping value must be non-negative, got {self.clamping!r}")

        self.use_dummies = _require_bool(data, "use_dummies")
        self.non_negative_weights = _require_bool(data, "non_negative_weights")

