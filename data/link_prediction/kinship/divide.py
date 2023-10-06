from argparse import ArgumentParser
from random import randint
from os.path import splitext

'''This method takes a single 'input' file representing a set of triples,
and separates it into two files according to the proportion 1:X. '''

def split_1_to_k(file, number):
    # Read the corresponding file
    inputFile = open(file, "r") 
    Lines = inputFile.readlines()

    # Mine name and extension of original input file
    fname = splitext(file)[0]
    fext = splitext(file)[1]  

    # Define output files: two for training, and two for testing 
    smallOutput = open(fname + "_facts" + fext, "w")
    bigOutput = open(fname + "_graph" + fext, "w")

    #Initialise a random variable determining if the fact will go to small or big set,
    # with probability 1:X
    randomVariable = 0

    #For each read line, roll the random variable, and write in the corresponding file.
    for line in Lines:

        randomVariable = randint(0,int(number))

        if randomVariable == 0 :
            smallOutput.write(line)
        else :
            bigOutput.write(line)

    smallOutput.close
    bigOutput.close


if __name__ == '__main__':

    # Read the argument from command line 
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("X")
    args = parser.parse_args()
   
    # Execute split on read arguments
    split_1_to_k(args.input,args.X)


