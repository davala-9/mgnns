import rdflib as rdf
import gzip
from random import randint

type_predicate = rdf.term.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

# Target relations 
employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")

g = rdf.Graph()
g.parse('./aifb_fixed_complete.n3', format='n3')

g_train = rdf.Graph()
g_valid = rdf.Graph()
g_valid_neg = rdf.Graph()
g_valid_pos = rdf.Graph()
g_text = rdf.Graph()
g_test_neg = rdf.Graph()
g_test_pos = rdf.Graph()

binary_predicates = set()
unary_predicates = set()
target_classes = set()

for (s, p, o) in g:
    if p==type_predicate:
        unary_predicates.add(o)
    if p==employs:
        target_classes.add(s) 
        unary_predicates.add(s) 
        g.remove((s,p,o)) 
        randomVariable = randint(0,12)
        if randomVariable < 9:
            g_train.add((o,type_predicate,s))
        elif randomVariable > 8 and randomVariable < 12:
            g_test.add((o,type_predicate,s))
            g_test_pos.add((o,type_predicate,s))
        else:
            g_valid.add((o,type_predicate,s))
            g_valid_pos.add((o,type_predicate,s))
    if p==affiliation:
        target_classes.add(o)
        unary_predicates.add(o) 
        g.remove((s,p,o)) 
        randomVariable = randint(0,12)
        if randomVariable < 9:
            g_train.add((s,type_predicate,o))
        elif randomVariable > 8 and randomVariable < 12:
            g_test.add((s,type_predicate,o))
            g_test_pos.add((s,type_predicate,o))
        else:
            g_valid.add((s,type_predicate,o))
            g_valid_pos.add((s,type_predicate,o))
    else:
        binary_predicates.add(p)

with gzip.open('aifb_graph.nt.gz', 'wb') as output:
    g.serialize(output, format='nt')
g.close()

with gzip.open('aifb_train.nt.gz', 'wb') as output:
    g_train.serialize(output, format='nt')
g_train.close()

with gzip.open('aifb_valid_pos.nt.gz', 'wb') as output:
    g_valid_pos.serialize(output, format='nt')
g_valid_pos.close()

with gzip.open('aifb_test_pos.nt.gz', 'wb') as output:
    g_test_pos.serialize(output, format='nt')
g_test_pos.close()

target_classes = list(target_classes)
assert(len(target_classes) > 1)

for (s, p, o) in g_valid_pos:
    i = randint(0,len(target_classes)-1)
    while target_classes[i] == o:
        i = randint(0,len(target_classes)-1)
    g_valid_neg.add((s,p,target_classes[i])) 
    g_valid.add((s,p,target_classes[i])) 

for (s, p, o) in g_test_pos:
    i = randint(0,len(target_classes)-1)
    while target_classes[i] == o:
        i = randint(0,len(target_classes)-1)
    g_test_neg.add((s,p,target_classes[i])) 
    g_test.add((s,p,target_classes[i])) 

with gzip.open('aifb_valid_neg.nt.gz', 'wb') as output:
    g_valid_neg.serialize(output, format='nt')
g_valid_neg.close()

with gzip.open('aifb_test_neg.nt.gz', 'wb') as output:
    g_test_neg.serialize(output, format='nt')
g_test_neg.close()

with gzip.open('aifb_valid.nt.gz', 'wb') as output:
    g_valid.serialize(output, format='nt')
g_valid.close()

with gzip.open('aifb_test.nt.gz', 'wb') as output:
    g_test.serialize(output, format='nt')
g_test.close()


output_file_name = 'predicates.csv'
outputFile = open(output_file_name, "w")
for bp in unary_predicates:
    outputFile.write(bp + ',1' + '\n')
for up in binary_predicates:
    outputFile.write(up + ',2' + '\n')
outputFile.close()
