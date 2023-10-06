from argparse import ArgumentParser
from os.path import splitext
import random
import os
import math

#  The incomplete graph is chosen to a) know the signature, and b) ensure that facts in the original graph are not chosen as negative examples. 
def generate_negative_examples(incomplete_graph_file, positive_examples_file, neg_examples_per_positive, output):
    
    constants = set()
    relations = set()
    classes = set()
    
    # Process file of positive facts     
    positive_examples = set()
    for line in open(positive_examples_file, "r").readlines():
        ent1, ent2, ent3 = line.split()
        if ent3.endswith('\n'):
            ent3 = ent3[:-1]
        read_triple = (ent1, ent2, ent3)
        if read_triple not in positive_examples:
            positive_examples.add(read_triple)
        if ent1 not in constants:
            constants.add(ent1)
        if ent2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            classes.add(ent3)
        else:
            constants.add(ent3)
            relations.add(ent2)
    print("Total positive examples read: {}".format(len(positive_examples))) 
    
    # Total number of negative examples needed
    n_examples = int(neg_examples_per_positive)*len(positive_examples)
    print("Trying to generate {} negative examples...".format(n_examples))

    #  Process incomplete graph file 
    # True known facts is the union of the incomplete graph and the positive examples 
    true_known_facts = positive_examples.copy() 
    for line in open(incomplete_graph_file, "r").readlines():
        ent1, ent2, ent3 = line.split()
        if ent3.endswith('\n'):
            ent3 = ent3[:-1]
        read_triple = (ent1, ent2, ent3)
        if read_triple not in positive_examples:
            true_known_facts.add(read_triple)
        if ent1 not in constants:
            constants.add(ent1)
        if ent2 == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>":
            classes.add(ent3)
        else:
            constants.add(ent3)
            relations.add(ent2)

    # Convert sets to list in order to sample
    constants = list(constants)
    relations = list(relations)

    negative_examples = set()
    visible_counter = 0 
    checkpoint = math.floor(n_examples/10) 

    while len(negative_examples) < n_examples:
        e_head = random.sample(constants,1)[0]
        e_rel  = random.sample(relations,1)[0]
        e_tail = random.sample(constants,1)[0]
        fact = (e_head,e_rel,e_tail)
        if fact not in set.union(true_known_facts,negative_examples):
            negative_examples.add(fact)
            visible_counter += 1
            if visible_counter % checkpoint == 0:
                print("Found {} negative examples so far.".format(len(negative_examples)))
    
    print("All negative examples found.")

    #  Print to output file
    output_file = open(output, "w") 
    pe = iter(positive_examples)
    ne = iter(negative_examples)
    for fact in pe: 
        (ent1, ent2, ent3) = fact
        output_file.write(ent1 + '\t' + ent2 + '\t' + ent3 + '\t' + "1" + '\n')
        for i in range(0,int(neg_examples_per_positive)): 
            (ent1, ent2, ent3) = next(ne)
            output_file.write(ent1 + '\t' + ent2 + '\t' + ent3 + '\t' + "0" + '\n')
   
    
# Read the argument from command line

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--positive-examples',
            help='Name of the file with the positive examples')
    parser.add_argument('--incomplete-graph',
            help='Name of the file with the original incomplete graph')
    parser.add_argument('--num-examples',help = "Number of negative examples per positive")
    parser.add_argument('--name-output', help = "Name of the folder where the examples are generated")
    args = parser.parse_args()
    generate_negative_examples(args.incomplete_graph,args.positive_examples,args.num_examples,args.name_output)

