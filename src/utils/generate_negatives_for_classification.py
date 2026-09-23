# Generates negative examples for node classification: for each positive example "entity rdf:type class", either
# one random other class, or (with --all-negatives) every other class. The possible classes are read from an
# R-GCN-format file (a header line, then "entity\tid\tclass" lines).
import argparse
from random import randint

from src.utils.utils import TYPE_PRED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--complete', required=True, help='Complete dataset, so that we can extract the classes.')
    parser.add_argument('--positives', required=True, help='Dataset with the positive examples.')
    parser.add_argument('--output-negatives', required=True, help='Generated dataset with the negative examples.')
    parser.add_argument('--output-all', required=True,
                        help='Generated dataset with all examples (both positive and negative).')
    parser.add_argument('--all-negatives', action='store_true',
                        help='Optional argument. Gives all possible negative examples, instead of one per positive example.')
    args = parser.parse_args()

    possible_classes = set()
    with open(args.complete, 'r') as complete_file:
        next(complete_file, None)  # Skip the header line
        for line in complete_file:
            s, p, o = line.split()
            possible_classes.add(o)
    possible_classes = list(possible_classes)

    with open(args.positives, 'r') as examples_file, open(args.output_all, 'w') as output_all, \
            open(args.output_negatives, 'w') as output_neg:
        for line in examples_file:
            s, p, o = line.split()
            assert p == TYPE_PRED, "Positive example appears to not be using the type predicate"
            assert o in possible_classes, "Positive example appears to mention a class not from the signature"
            output_all.write("{}\t{}\t{}\n".format(s, TYPE_PRED, o))
            if args.all_negatives:
                negative_classes = [klass for klass in possible_classes if klass != o]
            else:
                i = randint(0, len(possible_classes) - 1)
                while possible_classes[i] == o:
                    i = randint(0, len(possible_classes) - 1)
                negative_classes = [possible_classes[i]]
            for klass in negative_classes:
                output_all.write("{}\t{}\t{}\n".format(s, TYPE_PRED, klass))
                output_neg.write("{}\t{}\t{}\n".format(s, TYPE_PRED, klass))
