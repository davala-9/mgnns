# Converts an examples file in the R-GCN format (a header line, then "entity\tid\tclass" lines) into a tsv file
# of unary facts "entity\trdf:type\tclass".
import argparse

from src.utils.utils import TYPE_PRED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Name of the input file.')
    parser.add_argument('--output', required=True, help='Name of the output file.')
    args = parser.parse_args()

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        next(input_file, None)  # Skip the header line
        for line in input_file:
            s, p, o = line.split()
            output_file.write("{}\t{}\t{}\n".format(s, TYPE_PRED, o))
