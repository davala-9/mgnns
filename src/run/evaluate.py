import torch
import argparse

from src.rule_extraction.full_program import EquivalentProgramExtractor
from src.utils.data_parser import parse
from src.config.config import threshold_argument
from src.model.gnn_transformation import apply_gnn_transformation
from src.utils.utils import check
from src.run.compute_metrics import compute_metrics, f1score, THRESHOLDS
from src.run.folders import create_experiment_folder, load_encoder, load_model, record
from src.rule_extraction.fact_explanation import FactExplainer
from src.model.cd_graph import TraceCollector

# Time (in seconds) the exhaustive rule search gets per head predicate before falling back to a Hail Mary.
PROGRAM_EXTRACTION_TIME_BUDGET_SECONDS = 5
# How many of the highest-scoring test predictions get an explanation rule.
N_FACTS_TO_EXPLAIN = 20

EXPERIMENT = "evaluate"

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help='Path of the model folder (created by src.run.train)')
    parser.add_argument("--threshold", required=True, type=threshold_argument,
                        help='Fact derivation threshold for validation and testing, between 0 and 1')
    parser.add_argument("--skip-program", action="store_true", help="Skip equivalent program extraction")
    return parser

# Validation
def validate(dd, external_encoder, internal_encoder, model, threshold, device, ef):
    print("Validating...")
    valid_graph_dataset = parse(check(dd / "valid_graph.tsv", "Validation graph"))
    predictions = apply_gnn_transformation(valid_graph_dataset, external_encoder, internal_encoder, model,
                                           threshold, device)  # Ignore activations, don't save.
    compute_metrics(predictions, dd / "valid_pos.tsv", dd / "valid_neg.tsv", ef / "valid_metrics.txt")
    # TODO: print_best_threshold

# Test
def test(dd, external_encoder, internal_encoder, model, threshold, device, ef):
    print("Testing...")
    test_graph_dataset = parse(check(dd / "test_graph.tsv", "Test graph"))
    trace = TraceCollector()
    predictions = apply_gnn_transformation(test_graph_dataset, external_encoder, internal_encoder, model,
                                           threshold, device, trace_collector=trace)
    compute_metrics(predictions, dd / "test_pos.tsv", dd / "test_neg.tsv", ef / "test_metrics.txt")
    return predictions, test_graph_dataset, trace

# The threshold (among those reported in the metrics file) with the highest F1 score over the given examples;
# ties go to the smallest such threshold.
def optimal_threshold(predictions, positive_examples, negative_examples):
    positive_scores = [predictions.get(fact, 0) for fact in parse(check(positive_examples, "Positive examples"))]
    negative_scores = [predictions.get(fact, 0) for fact in parse(check(negative_examples, "Negative examples"))]
    def f1_at(threshold):
        tp = sum(score > threshold for score in positive_scores)
        fp = sum(score > threshold for score in negative_scores)
        return f1score(tp, fp, len(positive_scores) - tp)
    return max(THRESHOLDS, key=f1_at)

def save_predictions(ef, predictions):
    derivations_file = ef / "predicted_triples.tsv"
    derivations_file_scored = ef / "predicted_triples_scored.tsv"
    to_print = []
    for (s, p, o) in predictions:
        to_print.append((predictions[s, p, o], (s, p, o)))
    to_print = sorted(to_print, reverse=True)  # Print from the fact with the highest score to the least
    with open(derivations_file, 'w') as output:
        for (score, (s, p, o)) in to_print:
            output.write("{}\t{}\t{}\n".format(s, p, o))
    with open(derivations_file_scored, 'w') as output2:
        for (score, (s, p, o)) in to_print:
            output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))

def extract_program(ef, device, model, threshold, external_encoder, internal_encoder, predicate=None):
    print(f"Computing full equivalent program with threshold {threshold}...")
    program_file = ef / "program.txt"
    program_extractor = EquivalentProgramExtractor(device, model, threshold, external_encoder, internal_encoder)
    predicate_positions = None
    if predicate is not None:
        predicate_positions = [program_extractor.resolve_predicate_position(predicate)]
    program_extractor.compute_all_upper_bounds(predicate_positions)
    program_extractor.get_all_rules(program_file, PROGRAM_EXTRACTION_TIME_BUDGET_SECONDS, predicate_positions)

def explain_facts(ef,predictions,device,model,threshold,trace,external_encoder, internal_encoder, test_graph_dataset):
    print(f"Computing prediction explanations with threshold {threshold}...")
    explanations_file = ef / "explanations.txt"
    sorted_predictions = sorted((fact for fact in predictions if predictions[fact] > threshold),
                                key=predictions.get, reverse=True)
    explainer = FactExplainer(device, model, threshold, trace, external_encoder, internal_encoder,
                              test_graph_dataset)
    with open(explanations_file, 'w') as output:
        for fact in sorted_predictions[:N_FACTS_TO_EXPLAIN]:
            rule = explainer.explain_fact(fact)
            output.write("{}\n".format(fact))
            output.write(rule + '\n\n')

# Evaluates a trained model on the validation and test data, extracts a program from it, and explains its top-scoring
# test predictions; program and explanations use the threshold that maximises F1 over the test data.
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    cfg, exp_folder = create_experiment_folder(EXPERIMENT, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    external_encoder, internal_encoder = load_encoder(args.model, cfg.encoding_scheme)
    model = load_model(args.model, device)
    validate(cfg.data_dir, external_encoder, internal_encoder, model, args.threshold, device, exp_folder)
    predictions, test_graph_dataset, trace = test(cfg.data_dir, external_encoder, internal_encoder, model,
                                                  args.threshold, device, exp_folder)
    save_predictions(exp_folder, predictions)
    extraction_threshold = optimal_threshold(predictions, cfg.data_dir / "test_pos.tsv", cfg.data_dir / "test_neg.tsv")
    record(exp_folder, "extraction_threshold", extraction_threshold)
    if not args.skip_program:
        extract_program(exp_folder, device, model, extraction_threshold, external_encoder, internal_encoder)
    explain_facts(exp_folder, predictions, device, model, extraction_threshold, trace, external_encoder,
                  internal_encoder, test_graph_dataset)

# TODO: Separate responsibilities better in the test method.