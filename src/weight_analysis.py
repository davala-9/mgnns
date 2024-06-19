import torch
from torch_geometric.data import Data
from numpy import arange
from numpy import trapz
from numpy import nan_to_num
import argparse
import data_parser
import os.path
import rdflib as rdf
from encoding_schemes import CanonicalEncoderDecoder, ICLREncoderDecoder
from utils import load_predicates

parser = argparse.ArgumentParser(description="Evaluate a trained GNNs")
parser.add_argument('--load-model-name',
                    help='Filename of trained model to load')
parser.add_argument('--predicates',
                    help='File with the fixed, ordered list of predicates we consider.')
parser.add_argument('--output',
                    default=None,
                    help='Print the classification metrics.')
args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.load_model_name).to(device)

    for name, param in model.named_parameters():
        print("NAME : {}".format(name))
        print("PARAM : {}".format(param))

    # counter = {-10: 0, -9: 0, -8: 0, -7: 0, -6: 0, -5: 0, -4: 0, -3: 0, -2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    # for layer in range(1, model.num_layers + 1):
    #     for row in range(model.matrix_A(2).size()[0]):
    #         for column in range(model.matrix_A(2).size()[1]):
    #             print(model.matrix_A(2)[row][column] == 0)
    #             print("BEFORE: {}".format(model.matrix_A(2)[row][column].item()))
    #             if model.matrix_A(2)[row][column].item() < 0.001:
    #                 model.matrix_A(2)[row][column] = 0
    #             print("AFTER: {}".format(model.matrix_A(2)[row][column].item()))

        # for colour in range(-1, 4):
        #     print("counting matrix for layer {} and colour {}".format(layer, colour))
        #     if colour == -1:
        #         w_matrix = model.matrix_A(layer)
        #     else:
        #         w_matrix = model.matrix_B(layer, colour)
        #     for row in range(w_matrix.size()[0]):
        #         for column in range(w_matrix.size()[1]):
        #             element = w_matrix[row][column].item()
        #             if element < pow(10,-3):
                    # if element < pow(10, -10):
                    #     counter[-10] += 1
                    # elif element < pow(10, -9):
                    #     counter[-9] += 1
                    # elif element < pow(10, -8):
                    #     counter[-8] += 1
                    # elif element < pow(10, -7):
                    #     counter[-7] += 1
                    # elif element < pow(10, -6):
                    #     counter[-6] += 1
                    # elif element < pow(10, -5):
                    #     counter[-5] += 1
                    # elif element < pow(10, -4):
                    #     counter[-4] += 1
                    # elif element < pow(10, -3):
                    #     counter[-3] += 1
                    # elif element < pow(10, -2):
                    #     counter[-2] += 1
                    # elif element < pow(10, -1):
                    #     counter[-1] += 1
                    # elif element < pow(10, 0):
                    #     counter[0] += 1
                    # elif element < pow(10, 1):
                    #     counter[1] += 1
                    # elif element < pow(10, 2):
                    #     counter[2] += 1
                    # else:
                    #     counter[3] += 1

    print(counter)

