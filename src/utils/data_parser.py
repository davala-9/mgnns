import rdflib as rdf
from pathlib import Path

def parse_as_nt(file):
    dataset = []
    graph = rdf.Graph()
    graph.parse(file, format='nt')
    for s, p, o in graph:
        dataset.append((str(s), str(p), str(o)))
    return dataset


def parse_as_tsv(file):
    with open(file, "r") as inputfile:
        lines = inputfile.readlines()
    dataset = []
    for line in lines:
        ent1, ent2, ent3 = line.split()
        dataset.append((ent1, ent2, ent3))
    return dataset


def parse(file: Path):
    if file.suffix == '.nt':
        return parse_as_nt(file)
    elif file.suffix == '.tsv':
        return parse_as_tsv(file)
    else:
        print("Error, data format not supported. Use .nt or .tsv")
        return None