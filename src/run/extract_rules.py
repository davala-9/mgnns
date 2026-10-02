# This script should load a given model and encoders, and then run rule extraction on them.
import argparse
import torch

from src.config.config import threshold_argument
from src.run.evaluate import extract_program
from src.run.folders import create_experiment_folder, load_encoder, load_model

EXPERIMENT = "extract-rules"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help='Path of the model folder (created by src.run.train)')
    parser.add_argument("--threshold", required=True, type=threshold_argument,
                        help='Fact derivation threshold, between 0 and 1')
    parser.add_argument("--predicate", help='Only extract rules whose head is this predicate. '
                                             'If omitted, extracts rules for all predicates.')
    args = parser.parse_args()

    cfg, ef = create_experiment_folder(EXPERIMENT, args.model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    external_encoder, internal_encoder = load_encoder(args.model, cfg.encoding_scheme)

    extract_program(ef, device, model, args.threshold, external_encoder, internal_encoder, args.predicate)
