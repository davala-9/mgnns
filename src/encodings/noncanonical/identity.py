from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from src.utils.utils import TYPE_PRED

class IdentityEncoderDecoder(NonCanonicalEncoder):

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_unary_predicates = []
        self.canonical_binary_predicates = []
        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                predicate, _, arity = line.split() # predicates are duplicate so we ignore the second
                arity = int(arity)
                if arity == 1:
                    self.canonical_unary_predicates.append(predicate)
                else:
                    self.canonical_binary_predicates.append(predicate)
        else:
            self.canonical_binary_predicates = binary_predicates
            self.canonical_unary_predicates = unary_predicates

    def encode_dataset(self, dataset, **kwargs):
        return dataset

    def decode_dataset(self, dataset):
        return dataset

    def decode_fact(self, s, p, o):
        return s, p, o

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for predicate in self.canonical_unary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 1))
        for predicate in self.canonical_binary_predicates:
            output.write("{}\t{}\t{}\n".format(predicate, predicate, 2))
        output.close()

    def get_canonical_equivalent(self, fact):
        s, p, o = fact
        return s, p, o

    def unfold(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder,**kwargs):

        data_conj = []  # Not necessarily tree-shaped

        # Data variable list
        data_var_prefix = "X"
        data_var_counter = 0
        root_variables = [data_var_prefix + str(data_var_counter)]

        def new_variable():
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        if can_conj.is_empty():
            return data_conj, root_variables   # Return the empty conjunction if the tree-shaped conjunction is empty.

        def unfold_variable(can_var: Variable, data_var: str):
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_conj.append((data_var, TYPE_PRED, can_predicate)) # canonical predicate is data predicate
            for (_, col, _), child_var in can_var.children.items():
                new_data_var = new_variable()
                bin_predicate = internal_encoder.binary_pred_colour_dict.inverse[col]
                data_conj.append((new_data_var, bin_predicate, data_var))
                unfold_variable(child_var, new_data_var)

        unfold_variable(can_conj.root_node, root_variables[0])

        return data_conj, root_variables