import argparse
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument('--complete',
                    help='Complete dataset, so that we can extract the classes.')
parser.add_argument('--positives',
                    help='Dataset with the positive examples.')
parser.add_argument('--output-negatives',
                    help='Generated dataset with the negative examples.')
parser.add_argument('--output-all',
                    help='Generated dataset with all examples (both positive and negative).')
parser.add_argument('--all-negatives',
                    default=None,
                    action='store_true', 
                    help='Optional argument. Gives all possible negative examples, instead of one per positive example.')
args = parser.parse_args()

type_pred =  "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

if __name__ == "__main__":

    possible_classes = set()

    complete_file = open(args.complete, 'r')
    first_line = True 
    for line in complete_file.readlines():
        if not first_line:
            s, p, o = line.split()
            possible_classes.add(o)
        if first_line:
            first_line = False
    complete_file.close()

    possible_classes = list(possible_classes)

    output_all = open(args.output_all, 'w')
    output_neg = open(args.output_negatives, 'w')

    examples_file = open(args.positives, 'r')
    for line in examples_file.readlines():
        s, p, o = line.split()
        assert p == type_pred, "Positive example appears to not be using the type predicate"
        assert o in possible_classes, "Positive example appears to mention a class not from the signature"
        output_all.write("{}\t{}\t{}\n".format(s, type_pred, o))
        if args.all_negatives:
            for klass in possible_classes:
                if klass != o:
                    output_all.write("{}\t{}\t{}\n".format(s, type_pred, klass))
                    output_neg.write("{}\t{}\t{}\n".format(s, type_pred, klass))

        else:
            i = randint(0,len(possible_classes)-1) 
            while possible_classes[i] == o:
                i = randint(0,len(possible_classes)-1)
            output_all.write("{}\t{}\t{}\n".format(s, type_pred, possible_classes[i]))
            output_neg.write("{}\t{}\t{}\n".format(s, type_pred, possible_classes[i]))
    output_all.close()
    output_neg.close()


