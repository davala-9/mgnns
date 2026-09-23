from src.encodings.noncanonical.noncanonical import VariableNamer


def test_variable_namer_counts_up_from_x0():
    namer = VariableNamer()
    assert [namer.new_variable() for _ in range(3)] == ["X0", "X1", "X2"]


def test_variable_namers_are_independent():
    first, second = VariableNamer(), VariableNamer()
    first.new_variable()
    assert second.new_variable() == "X0"
