import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    help='Name of the input file.')
parser.add_argument('--output',
                    help='Name of the output file.')
args = parser.parse_args()

type_pred =  "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

if __name__ == "__main__":

    output_file = open(args.output, 'w')
    
    input_file = open(args.input, 'r')
    first_line = True 
    for line in input_file.readlines():
        if not first_line:
            s, p, o = line.split()
            output_file.write("{}\t{}\t{}\n".format(s, type_pred, o))
        if first_line:
            first_line = False
    input_file.close()
    output_file.close()

