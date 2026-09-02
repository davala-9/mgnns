from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder, GroundContext
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

    def unary_can_predicate_to_data_predicate(self, predicate:str):
        return predicate

    def unary_can_predicate_to_data_predicate_arity(self, predicate:str):
        return 1

    def unfold_all(self, can_conj:TreeShapedConjunction, internal_encoder:CanonicalEncoderDecoder, head_predicate: str):

        data_conj = []  # Not necessarily tree-shaped

        # Data variable list
        data_var_prefix = "X"
        data_var_counter = 0
        root_variable = data_var_prefix + str(data_var_counter)

        def new_variable():
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        var_id_to_datavar = {0: root_variable}
        for var_id in range(len(can_conj)):
            for feat in can_conj.features[var_id]:
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_conj.append((var_id_to_datavar[var_id], TYPE_PRED, can_predicate)) # canonical pred is data pred
            for (_, col, _), child_id in can_conj.children[var_id].items():
                var_id_to_datavar = {child_id: new_variable()}
                bin_predicate = internal_encoder.binary_pred_colour_dict.inverse[col]
                data_conj.append((var_id_to_datavar[child_id], bin_predicate, var_id_to_datavar[var_id])) # This order
        head = (root_variable, TYPE_PRED, head_predicate)

        return [data_conj], head

    def unfold_match_ground(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder,
               head_predicate: str, grounding_context: GroundContext):

        # The unfolding is unique so grounding_context can be safely ignored
        data_conj_set, head = self.unfold_all(can_conj, internal_encoder, head_predicate)
        (data_conj,) = data_conj_set
        return data_conj, head



