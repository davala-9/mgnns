# Writes a predicates.csv file (lines "predicate,arity") listing every predicate in an .nt graph, plus every class
# mentioned in an R-GCN-format examples file (a header line, then "entity\tid\tclass" lines).
import argparse

import rdflib as rdf

from src.utils.utils import TYPE_PRED

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True, help='Name of the full graph file.')
    parser.add_argument('--examples', required=True, help='Name of the file with both training and test examples')
    parser.add_argument('--output', required=True, help='Name of the output file.')
    args = parser.parse_args()

    type_predicate = rdf.term.URIRef(TYPE_PRED)
    binary_predicates = set()
    unary_predicates = set()

    print("Reading graph...")
    g = rdf.Graph()
    g.parse(args.graph, format='nt')
    for (s, p, o) in g:
        if p == type_predicate:
            unary_predicates.add(str(o))
        else:
            binary_predicates.add(str(p))

    print("Reading examples...")
    with open(args.examples, 'r') as examples_file:
        next(examples_file, None)  # Skip the header line
        for line in examples_file:
            s, p, o = line.split()
            unary_predicates.add(o)

    print("Writing predicates...")
    with open(args.output, "w") as output_file:
        for up in unary_predicates:
            output_file.write(up + ',1' + '\n')
        for bp in binary_predicates:
            output_file.write(bp + ',2' + '\n')
