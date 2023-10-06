import argparse
import rdflib as rdf

parser = argparse.ArgumentParser()
parser.add_argument('--graph',
                    help='Name of the full graph file.')
parser.add_argument('--examples',
                    help='Name of the file with both training and test examples')
parser.add_argument('--output',
                    help='Name of the output file.')
args = parser.parse_args()

type_predicate = rdf.term.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

if __name__ == "__main__":

    binary_predicates = set()
    unary_predicates = set()

    print("Reading graph...")

    g = rdf.Graph()
    g.parse(args.graph, format='nt')
    for (s, p, o) in g:
        if p==type_predicate:
            unary_predicates.add(o)
        else:
            binary_predicates.add(p)

    print("Reading examples...")

    examples_file = open(args.examples, 'r')
    first_line = True 
    for line in examples_file.readlines():
        if not first_line:
            s, p, o = line.split()
            unary_predicates.add(o)
        if first_line:
            first_line = False
    examples_file.close()

    print("Writing predicates...")

    output_file = open(args.output, "w")
    for up in unary_predicates:
        output_file.write(up + ',1' + '\n')
    for bp in binary_predicates:
        output_file.write(bp + ',2' + '\n')
    output_file.close()
    
