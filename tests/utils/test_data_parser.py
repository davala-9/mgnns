import pytest

from src.utils.data_parser import parse
from src.utils.utils import TYPE_PRED


def test_parse_tsv(tmp_path):
    data_file = tmp_path / "facts.tsv"
    data_file.write_text(f"a\tR\tb\na\t{TYPE_PRED}\tA\n")
    assert parse(data_file) == [("a", "R", "b"), ("a", TYPE_PRED, "A")]


def test_parse_unsupported_format_raises(tmp_path):
    data_file = tmp_path / "facts.csv"
    data_file.write_text("a,R,b\n")
    with pytest.raises(ValueError, match="not supported"):
        parse(data_file)
