# This script should load a given model and encoders, and then run rule extraction on them.
import argparse
import torch
from pathlib import Path

from src.config.config import EncoderType
from src.encodings.canonical import CanonicalEncoderDecoder
from src.run.run_experiment import load_model, load_encoder, extract_program

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help='Path of the folder where we have model & encoders')
    parser.add_argument("threshold", type=float, help='Fact derivation threshold')
    parser.add_argument("output", help='Path of the folder where we will save the output')
    parser.add_argument("--predicate", help='Only extract rules whose head is this predicate. '
                                             'If omitted, extracts rules for all predicates.')
    args = parser.parse_args()

    ef = Path(args.output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.input, device)
    encoder_scheme = EncoderType.CANONICAL
    external_encoder, internal_encoder = load_encoder(args.input, encoder_scheme)

    extract_program(ef,device,model,args.threshold,external_encoder,internal_encoder,args.predicate)



