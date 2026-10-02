from src.run.run_experiment import optimal_threshold


def write_examples(path, facts):
    path.write_text("".join(f"{s}\t{p}\t{o}\n" for (s, p, o) in facts))
    return path


def test_optimal_threshold_separates_positives_from_negatives(tmp_path):
    positives = write_examples(tmp_path / "pos.tsv", [("a", "R", "b"), ("b", "R", "c")])
    negatives = write_examples(tmp_path / "neg.tsv", [("a", "R", "c"), ("c", "R", "a")])
    predictions = {("a", "R", "b"): 0.9, ("b", "R", "c"): 0.5, ("a", "R", "c"): 0.3}
    # Every threshold in [0.3, 0.5) gives F1 1; ties go to the smallest.
    assert optimal_threshold(predictions, positives, negatives) == 0.3


def test_optimal_threshold_treats_missing_predictions_as_score_zero(tmp_path):
    positives = write_examples(tmp_path / "pos.tsv", [("a", "R", "b")])
    negatives = write_examples(tmp_path / "neg.tsv", [("a", "R", "c")])
    # The negative is never predicted, so even the smallest threshold is perfect.
    assert optimal_threshold({("a", "R", "b"): 0.001}, positives, negatives) == 0
