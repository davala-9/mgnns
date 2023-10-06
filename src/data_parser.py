import rdflib as rdf


def parse_as_nt(file):
    dataset = []
    graph = rdf.Graph()
    graph.parse(file, format='nt')
    for s, p, o in graph:
        dataset.append((str(s), str(p), str(o)))
    return dataset


def parse_as_tsv(file):
    inputfile = open(file, "r")
    lines = inputfile.readlines()
    dataset = []
    for line in lines:
        ent1, ent2, ent3 = line.split()
        dataset.append((ent1, ent2, ent3))
    return dataset


def parse(file):
    if file.endswith('.nt'):
        return parse_as_nt(file)
    elif file.endswith('.tsv'):
        return parse_as_tsv(file)
    else:
        print("Error, data format not supported. Use .nt or .tsv")
        return None