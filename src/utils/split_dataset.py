# One-off script that splits the AIFB dataset (./aifb_fixed_complete.n3) into train/validation/test files for
# node classification. The target facts are "X employs Y" and "Y affiliation X", both turned into the unary fact
# "Y rdf:type X" and removed from the graph; roughly 9/13 go to training, 3/13 to testing and 1/13 to validation.
# Each validation/test positive also gets one random negative with a different target class.
import gzip
from random import randint

import rdflib as rdf

from src.utils.utils import TYPE_PRED

INPUT_FILE = './aifb_fixed_complete.n3'

type_predicate = rdf.term.URIRef(TYPE_PRED)

# Target relations
employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")


# Adds `fact` to the training, test or validation graphs at random.
def add_to_random_split(fact, g_train, g_test, g_test_pos, g_valid, g_valid_pos):
    random_variable = randint(0, 12)
    if random_variable < 9:
        g_train.add(fact)
    elif random_variable < 12:
        g_test.add(fact)
        g_test_pos.add(fact)
    else:
        g_valid.add(fact)
        g_valid_pos.add(fact)


# For each positive (s, p, o) in g_pos, adds (s, p, o') with a random other target class o' to both g_neg and g_all.
def add_random_negatives(g_pos, g_neg, g_all, target_classes):
    for (s, p, o) in g_pos:
        i = randint(0, len(target_classes) - 1)
        while target_classes[i] == o:
            i = randint(0, len(target_classes) - 1)
        g_neg.add((s, p, target_classes[i]))
        g_all.add((s, p, target_classes[i]))


def serialize_gzipped(graph, file_name):
    with gzip.open(file_name, 'wb') as output:
        graph.serialize(output, format='nt')
    graph.close()


if __name__ == "__main__":
    g = rdf.Graph()
    g.parse(INPUT_FILE, format='n3')

    g_train = rdf.Graph()
    g_valid = rdf.Graph()
    g_valid_neg = rdf.Graph()
    g_valid_pos = rdf.Graph()
    g_test = rdf.Graph()
    g_test_neg = rdf.Graph()
    g_test_pos = rdf.Graph()
    splits = (g_train, g_test, g_test_pos, g_valid, g_valid_pos)

    binary_predicates = set()
    unary_predicates = set()
    target_classes = set()

    for (s, p, o) in list(g):  # A copy, since target facts are removed from g as we go
        if p == type_predicate:
            unary_predicates.add(o)
        elif p == employs:
            target_classes.add(s)
            unary_predicates.add(s)
            g.remove((s, p, o))
            add_to_random_split((o, type_predicate, s), *splits)
        elif p == affiliation:
            target_classes.add(o)
            unary_predicates.add(o)
            g.remove((s, p, o))
            add_to_random_split((s, type_predicate, o), *splits)
        else:
            binary_predicates.add(p)

    target_classes = list(target_classes)
    assert len(target_classes) > 1
    add_random_negatives(g_valid_pos, g_valid_neg, g_valid, target_classes)
    add_random_negatives(g_test_pos, g_test_neg, g_test, target_classes)

    serialize_gzipped(g, 'aifb_graph.nt.gz')
    serialize_gzipped(g_train, 'aifb_train.nt.gz')
    serialize_gzipped(g_valid_pos, 'aifb_valid_pos.nt.gz')
    serialize_gzipped(g_test_pos, 'aifb_test_pos.nt.gz')
    serialize_gzipped(g_valid_neg, 'aifb_valid_neg.nt.gz')
    serialize_gzipped(g_test_neg, 'aifb_test_neg.nt.gz')
    serialize_gzipped(g_valid, 'aifb_valid.nt.gz')
    serialize_gzipped(g_test, 'aifb_test.nt.gz')

    with open('predicates.csv', "w") as output_file:
        for up in unary_predicates:
            output_file.write(up + ',1' + '\n')
        for bp in binary_predicates:
            output_file.write(bp + ',2' + '\n')
