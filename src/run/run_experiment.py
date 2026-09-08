from base64 import encode

import torch
from torch_geometric.data import Data, DataLoader
import argparse

from src.rule_extraction.full_program import EquivalentProgramExtractor
from src.utils.data_parser import parse
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.config.config import EncoderType, ExperimentConfig
from src.model.gnn_transformation import apply_gnn_transformation
from src.utils.utils import check, load_predicates
from src.run.train import train
from src.run.compute_metrics import compute_metrics
from src.rule_extraction.fact_explanation import FactExplainer
from src.model.gnn_architectures import GNN
from src.model.cd_graph import CDGraph, TraceCollector
from datetime import datetime
from pathlib import Path
import shutil

# Create parser to parse all arguments of the script
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help='Path of the configuration file controlling the experiment.')
    parser.add_argument("--load-model", help='Use existing model')
    parser.add_argument("--minimal", action="store_true", help="Minimise explanatory rules")
    parser.add_argument("--skip-program", action="store_true", help="Skip equivalent program extraction")
    return parser

# Read experiment configuration and create experiment folder, setting up experiment variables too.
def setup_experiment(args):
    cfg = ExperimentConfig(check(args.config_file, "configuration"))
    experiment_name = f"{cfg.data_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_folder = cfg.exp_dir / experiment_name
    experiment_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_file, experiment_folder)  # Copy configuration into experiment folder
    return cfg, experiment_folder

def load_encoder(path_name, encoding_scheme):
    print("Loading encoder from file...")
    load_ef = Path(path_name)
    internal_encoder = CanonicalEncoderDecoder(check(load_ef / 'internal_encoder.tsv', "Internal encoding"))
    if encoding_scheme == EncoderType.ICLR22:
        external_encoder = ICLREncoderDecoder(check(load_ef / 'external_encoder.tsv', "External encoding"))
    else:
        external_encoder = IdentityEncoderDecoder(check(load_ef / 'external_encoder.tsv', "External encoding"))
    return external_encoder, internal_encoder

def setup_encoder(cfg: ExperimentConfig, exp_folder: Path, encoder_pathname: str | None):
    if encoder_pathname:
       external_encoder, internal_encoder = load_encoder(encoder_pathname, cfg.encoding_scheme)
    else:
        print("Creating Encoder-Decoders...")
        # Set-up Encoder-Decoder
        dbp, dup = load_predicates(check(cfg.data_dir  / "predicates.csv", "Predicates"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(load_from_document=None,
                                                  unary_predicates=dup,
                                                  binary_predicates=dbp)
        else: # Default external encoding is the IdentityEncoder, which encodes each fact as itself
            external_encoder = IdentityEncoderDecoder(load_from_document=None,
                                                      unary_predicates=dup,
                                                      binary_predicates=dbp)
        internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
                                                   unary_predicates=external_encoder.canonical_unary_predicates,
                                                   binary_predicates=external_encoder.canonical_binary_predicates)
        external_encoder.save_to_file(exp_folder / 'external_encoder.tsv')
        internal_encoder.save_to_file(exp_folder / 'internal_encoder.tsv')
    return external_encoder, internal_encoder

def load_model(model_pathname, device):
    load_ef = Path(model_pathname)
    return torch.load(check(load_ef / "model.pt", "Model"), weights_only=False, map_location=device).to(device)

def setup_model(cfg: ExperimentConfig, exp_folder, device, ext_enc, int_enc, with_model: str | None):
    # Training (or Model Loading, if training is skipped)
    if with_model:
        model = load_model(with_model, device)
    else:
        print("Training...")
        # Load & encode training data
        # TODO: sanity check - warn if the training data contains any predicates out of the signature.
        graph_dataset = parse(check(cfg.data_dir / "train_graph.tsv", "Training graph"))
        cd_dataset = ext_enc.encode_dataset(graph_dataset, use_dummy_constants=cfg.use_dummies)
        cd_graph = int_enc.encode_dataset(cd_dataset)
        train_examples_dataset = parse(check(cfg.data_dir / "train_pos.tsv", "Training positive examples"))
        cd_train_examples = ext_enc.encode_dataset(train_examples_dataset)
        model = GNN(feature_dimension=cd_graph.delta, num_edge_colours=cd_graph.col_size,
                    aggregation_1=cfg.agg_function_1, aggregation_2=cfg.agg_function_2).to(device)
        train(cfg=cfg, device=device, internal_encoder=int_enc, model=model, cd_graph=cd_graph,
              train_examples=cd_train_examples, experiment_folder=exp_folder)
    return model

# Validation
def validate(dd, external_encoder, internal_encoder, model, exp_cfg, device, ef):
    print("Validating...")
    valid_graph_dataset = parse(check(dd / "valid_graph.tsv", "Validation graph"))
    predictions = apply_gnn_transformation(valid_graph_dataset, external_encoder, internal_encoder, model,
                                           exp_cfg.derivation_threshold, device)  # Ignore activations, don't save.
    compute_metrics(predictions, dd / "valid_pos.tsv", dd / "valid_neg.tsv", ef / "valid_metrics.txt")
    # TODO: print_best_threshold

# Test
def test(dd, external_encoder, internal_encoder, model, exp_cfg, device, ef):
    print("Testing...")
    test_graph_dataset = parse(check(dd / "test_graph.tsv", "Test graph"))
    trace = TraceCollector()
    predictions = apply_gnn_transformation(test_graph_dataset, external_encoder, internal_encoder, model,
                                           exp_cfg.derivation_threshold, device, trace_collector=trace)
    compute_metrics(predictions, dd / "test_pos.tsv", dd / "test_neg.tsv", ef / "test_metrics.txt")
    return predictions, test_graph_dataset, trace

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
    output.close()

def extract_program(ef, device, model, threshold, external_encoder, internal_encoder):
    print("Computing full equivalent program...")
    program_file = ef / "program.txt"
    program_extractor = EquivalentProgramExtractor(device, model, threshold, external_encoder,
                                                   internal_encoder)
    program_extractor.compute_all_upper_bounds()
    program_extractor.get_all_rules(program_file, 30)

def explain_facts(ef,predictions,device,model,cfg,trace,external_encoder, internal_encoder, test_graph_dataset):
    print("Computing prediction explanations...")
    explanations_file = ef / "explanations.txt"
    sorted_predictions = sorted(predictions, key=predictions.get, reverse=True)
    explainer = FactExplainer(device, model, cfg.derivation_threshold, trace, external_encoder, internal_encoder,
                              test_graph_dataset)
    with open(explanations_file, 'w') as output:
        for fact in sorted_predictions[:20]:  # TODO: replace magic number with parameter
            rule = explainer.explain_fact(fact)
            output.write("{}\n".format(fact))
            output.write(rule + '\n')
    output.close()

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    cfg, exp_folder = setup_experiment(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    external_encoder, internal_encoder = setup_encoder(cfg, exp_folder, args.load_model)
    model = setup_model(cfg, exp_folder, device, external_encoder, internal_encoder, args.load_model)
    validate(cfg.data_dir, external_encoder, internal_encoder, model, cfg, device, exp_folder)
    predictions, test_graph_dataset, trace = test(cfg.data_dir, external_encoder, internal_encoder, model, cfg, device,
                                                  exp_folder)
    save_predictions(exp_folder, predictions)
    if not args.skip_program:
        extract_program(exp_folder, device, model, cfg.derivation_threshold, external_encoder, internal_encoder)
    explain_facts(exp_folder,predictions,device,model,cfg,trace,external_encoder, internal_encoder, test_graph_dataset)

# TODO: Separate responsabilities better in the test method.