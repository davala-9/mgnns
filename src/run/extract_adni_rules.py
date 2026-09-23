# This script should load a given ADNI model and its internal encoder, and then run a greedy heuristic
# search over triangular matrices for one whose corresponding tree soundly derives "positive".
import argparse
from pathlib import Path

import torch

from src.adni.heuristic import POSITIVE_PREDICATE
from src.adni.matrix_search import greedy_climb, minimise, prune_isolated_nodes, value_order_table
from src.adni.matrix_to_tree import infer_dimensions, is_sound, matrix_to_tree
from src.adni.rule_printing import tree_to_rule
from src.adni.sparse_triangular_matrix import SparseTriangularMatrix
from src.encodings.canonical import CanonicalEncoderDecoder
from src.run.run_experiment import load_model
from src.utils.utils import check

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help='Path of the folder where we have the model & internal encoder')
    parser.add_argument("threshold", type=float, help='Fact derivation threshold')
    parser.add_argument("output", help='Path of the folder where we will save the output')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.input, device)
    internal_encoder = CanonicalEncoderDecoder(
        check(Path(args.input) / 'internal_encoder.tsv', "Internal encoding"))
    positive_position = internal_encoder.unary_pred_position_dict[POSITIVE_PREDICATE]
    d, max_value = infer_dimensions(internal_encoder)

    def check_soundness_with_nodes(matrix: SparseTriangularMatrix, included_nodes) -> bool:
        return is_sound(matrix, internal_encoder, model, device, positive_position, args.threshold, included_nodes)

    def check_soundness(matrix: SparseTriangularMatrix) -> bool:
        return is_sound(matrix, internal_encoder, model, device, positive_position, args.threshold)

    print(f"Searching over {d}x{d} triangular matrices (values 1..{max_value})...")
    value_order = value_order_table(d, max_value, internal_encoder, model)
    empty_matrix = SparseTriangularMatrix.empty(d, max_value)
    result = greedy_climb(empty_matrix, value_order, internal_encoder, model, check_soundness, verbose=True)

    output_folder = Path(args.output)
    output_folder.mkdir(parents=True, exist_ok=True)
    with open(output_folder / "program.txt", 'w') as output_file:
        if result is None:
            print("No sound matrix found.")
            output_file.write("No sound matrix found.\n")
        else:
            print("Minimising the sound matrix...")
            result = minimise(result, internal_encoder, model, check_soundness, verbose=True)
            print("Dropping isolated nodes...")
            included_nodes = prune_isolated_nodes(result, check_soundness_with_nodes, verbose=True)
            rule = tree_to_rule(
                matrix_to_tree(result, internal_encoder, included_nodes=included_nodes),
                internal_encoder, POSITIVE_PREDICATE)
            print(rule)
            output_file.write(rule + '\n')
