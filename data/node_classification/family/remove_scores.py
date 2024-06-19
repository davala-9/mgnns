from argparse import ArgumentParser
from os.path import splitext
import os

# Read the argument from command line
parser = ArgumentParser()
parser.add_argument('--input',help="Input file with facts and scores.")
parser.add_argument('--output',help="Output file with only the facts.")
args = parser.parse_args()

with open(args.input, "r") as input_file:
    with open(args.output, "w") as output_file:
        for line in input_file.readlines():
            output_file.write(line[:-3]+ '\n')
    output_file.close()
print("Wrote a copy of \n {}\n without the scores in file \n {}".format(args.input,args.output))

