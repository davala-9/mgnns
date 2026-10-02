import argparse
from pathlib import Path

import torch

from src.adni.matrix_search import greedy_climb, minimise, value_order_table
from src.adni.matrix_to_tree import is_sound
from src.adni.rule_printing import matrix_to_rule
from src.adni.signature import POSITIVE_PREDICATE, AdniSignature
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.config.config import threshold_argument
from src.run.folders import create_experiment_folder, load_model
from src.utils.utils import check

EXPERIMENT = "adni-rules"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help='Path of the model folder (created by src.run.train)')
    parser.add_argument("--threshold", required=True, type=threshold_argument,
                        help='Fact derivation threshold, between 0 and 1')
    args = parser.parse_args()
    _, output_folder = create_experiment_folder(EXPERIMENT, args.model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model, device)
    internal_encoder = CanonicalEncoderDecoder(check(Path(args.model) / 'internal_encoder.tsv', "Internal encoding"))
    signature = AdniSignature(internal_encoder)
    positive_position = signature.positive_pos
    d, max_value = signature.d, signature.max_value

    def check_soundness(matrix: SparseTriangularMatrix) -> bool:
        return is_sound(matrix, signature, model, device, positive_position, args.threshold)

    print(f"Searching over {d}x{d} triangular matrices (values 1..{max_value})...")
    value_order = value_order_table(signature, model) # Order potential edge values by their heuristic value
    empty_matrix = SparseTriangularMatrix.empty(d, max_value)
    result = greedy_climb(empty_matrix, value_order, signature, model, check_soundness, verbose=True)

    with open(output_folder / "program.txt", 'w') as output_file:
        if result is None:
            print("No sound matrix found.")
            output_file.write("No sound matrix found.\n")
        else:
            print("Minimising the sound matrix...")
            result = minimise(result, signature, model, check_soundness, verbose=True)
            rule = matrix_to_rule(result, signature, POSITIVE_PREDICATE)
            print(rule)
            output_file.write(rule + '\n')
