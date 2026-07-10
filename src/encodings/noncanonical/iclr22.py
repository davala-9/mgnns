import torch
from sympy.physics.units import second
from sympy.strategies import canon

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder
from bidict import bidict

from src.rule_extraction.fact_explanation import FactExplainer
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from src.utils.utils import TYPE_PRED

class ICLREncoderDecoder(NonCanonicalEncoder):

    # Fresh predicates that correspond to colours c1, c2, c3, c4 in the paper. Abbreviations match paper names
    col1 = "binary-pred-1"
    col2 = "binary-pred-2"
    col3 = "binary-pred-3"
    col4 = "binary-pred-4"

    # This is a placeholder predicate used for unfoldings. It simply says "these two appear together in the dataset"
    TOP_PREDICATE = "top-pred"

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_binary_predicates = [self.col1, self.col2, self.col3, self.col4]
        self.input_predicate_to_unary_canonical_dict = bidict()
        self.input_predicate_to_arity = {}
        if load_from_document is not None:
            for line in open(load_from_document, 'r').readlines():
                input_predicate, canonical_predicate, arity = line.split()
                self.input_predicate_to_unary_canonical_dict[input_predicate] = canonical_predicate
                self.input_predicate_to_arity[input_predicate] = int(arity)
        else:
            assert set(unary_predicates).isdisjoint(set(binary_predicates)) # Sanity check
            for pred in unary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = pred
                self.input_predicate_to_arity[pred] = 1
            for pred in binary_predicates:
                self.input_predicate_to_unary_canonical_dict[pred] = "unary-for-{}".format(pred)
                self.input_predicate_to_arity[pred] = 2
        # Maps pairs of constants to a new single term
        self.pair_term_dict = bidict()
        self.canonical_unary_predicates = list(self.input_predicate_to_unary_canonical_dict.inverse.keys())

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for input_predicate in self.input_predicate_to_unary_canonical_dict:
            output.write("{}\t{}\t{}\n".format(input_predicate,
                                               self.input_predicate_to_unary_canonical_dict[input_predicate],
                                               self.input_predicate_to_arity[input_predicate]))
        output.close()

    def term_for_pair(self, pair):
        if pair not in self.pair_term_dict:
            self.pair_term_dict[pair] = "term-for-{}-{}".format(pair[0], pair[1])
        return self.pair_term_dict[pair]

    def encode_fact(self, fact):
        encoded_dataset = []
        s, p, o = fact
        if p == TYPE_PRED:
            encoded_dataset.append((s, p, self.input_predicate_to_unary_canonical_dict[o]))
        else:
            a = s    # We rename to the paper's notation to make the code easier to read and write
            b = o
            ab = self.term_for_pair((a, b))
            ba = self.term_for_pair((b, a))
            encoded_dataset.append((ab, TYPE_PRED, self.input_predicate_to_unary_canonical_dict[p]))
            encoded_dataset.extend([(a, self.col1, ab), (ab, self.col1, a), (b, self.col1, ba), (ba, self.col1, b)])
            encoded_dataset.extend([(b, self.col2, ab), (ab, self.col2, b), (a, self.col2, ba), (ba, self.col2, a)])
            encoded_dataset.extend([(ab, self.col3, ba), (ba, self.col3, ab)])
            encoded_dataset.extend([(a, self.col4, b), (b, self.col4, a)])
        return encoded_dataset

    # Returns a new dataset over the new signature
    def encode_dataset(self, dataset, use_dummy_constants=False):

        encoded_dataset = []
        for fact in dataset:
            encoded_dataset.extend(self.encode_fact(fact))

        if use_dummy_constants:
            # Extract all constants from encoded_dataset. Less efficient than doing it on-the-fly, but it's cleaner code
            constants = set()
            for s, p, o in encoded_dataset:
                constants.add(s)
                if p != TYPE_PRED:
                    constants.add(o)
            for a in ['#', '##']:
                for b in constants:
                    ab = self.term_for_pair((a, b))
                    ba = self.term_for_pair((b, a))
                    encoded_dataset.extend([(a, self.col1, ab), (ab, self.col1, a), (b, self.col1, ba), (ba, self.col1, b)])
                    encoded_dataset.extend([(b, self.col2, ab), (ab, self.col2, b), (a, self.col2, ba), (ba, self.col2, a)])
                    encoded_dataset.extend([(ab, self.col3, ba), (ba, self.col3, ab)])
                    encoded_dataset.extend([(a, self.col4, b), (b, self.col4, a)])

        return encoded_dataset

    def get_canonical_equivalent(self, fact):
        (s, p, o) = fact
        if p == TYPE_PRED:
            return s, p, self.input_predicate_to_unary_canonical_dict[o]
        else:
            ab = self.term_for_pair((s, o))
            return ab, TYPE_PRED, self.input_predicate_to_unary_canonical_dict[p]

    def decode_dataset(self, canonical_dataset):
        return {decoded for s, p, o in canonical_dataset
                if (decoded := self.decode_fact(s, p, o)) is not None}

    def decode_fact(self, s, p, o):
        # All facts in an encoded dataset are unary; binary facts should not be decoded.
        assert(p == TYPE_PRED)
        data_predicate = self.input_predicate_to_unary_canonical_dict.inverse[o]
        if s in self.pair_term_dict.values() and self.input_predicate_to_arity[data_predicate] == 2:
            # We filter out any facts with unary-canonical predicates that match *unary* data predicates for
            # nodes that represent pairs. Ideally no predictions would be generated, but if they are, we ignore them.
            a, b = self.pair_term_dict.inverse[s]
            return a, self.input_predicate_to_unary_canonical_dict.inverse[o], b
        elif self.input_predicate_to_arity[data_predicate] == 1:
            # We filter out any facts with unary-canonical predicates that match *binary* data predicates for
            # nodes that represent constants. Ideally no predictions would be generated, but if they are, ignore them.
            # The node should be for a single constant.
            return s, TYPE_PRED, self.input_predicate_to_unary_canonical_dict.inverse[o]

    def is_term_for_pair(self, const):
        return const in self.pair_term_dict.inverse

    # Maps a canonical predicate to the arity of the original predicate, either 1 or 2
    def original_arity(self, canonical_predicate):
        return self.input_predicate_to_arity[self.input_predicate_to_unary_canonical_dict.inverse[canonical_predicate]]

    # This function takes a tree-shaped conjunction expressed in the Canonical Signature and returns a conjunction in
    # the Data Signature.
    # We traverse the canonical conjunction unfolding as we go.
    # Note that we unfold mainly canonic unary atoms, (which turn into either unary or binary data atoms)
    # Canonical binary atoms are often superfluous, but in some cases require the addition of a TOP predicate fact
    # Returns the unfolded conjunction and a list of the variables in the head (might be one or two)
    def unfold(self, can_conj: TreeShapedConjunction,
               internal_encoder: CanonicalEncoderDecoder, **kwargs):
        var_const_idx = kwargs.get("var_const_idx")
        cd_graph = kwargs.get("cd_graph")
        head_predicate_arity = kwargs.get("head_predicate_arity")
        head_is_binary = head_predicate_arity == 2

        def get_data_constants_for_tree_variable(var: Variable):
            canonical_constant_index = var_const_idx[var]
            canonical_constant = cd_graph.node_names[canonical_constant_index]
            if canonical_constant in self.pair_term_dict.inverse:
                return list(self.pair_term_dict.inverse[canonical_constant])
            else:
                return list(canonical_constant)

        data_conj = [] # Not necessarily tree-shaped
        data_var_to_const = {} # Not strictly necessary but helps debug and simplifies dealing with top predicates

        # Data variable list
        data_var_prefix = "X"
        data_var_counter = 0

        def new_variable():
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        root_variables = [data_var_prefix + str(data_var_counter)]
        data_var_to_const[root_variables[0]] = get_data_constants_for_tree_variable(can_conj.root_node)[0]
        if head_is_binary:
            second_root_data_var = new_variable()
            root_variables.append(second_root_data_var)
            data_var_to_const[root_variables[1]] = get_data_constants_for_tree_variable(can_conj.root_node)[1]

        # Unfold unary canonical atom that unifies with a canonical constant for a pair of data_constants.
        def unfold_variable_for_pair(can_var: Variable, first_data_var: str=None, second_data_var: str=None):
            # First, add the relevant atoms.
            assert first_data_var is not None or second_data_var is not None # We should know at least one of them.
            if first_data_var is None:
                first_data_var = new_variable()
                data_var_to_const[first_data_var] = get_data_constants_for_tree_variable(can_var)[0]
            if second_data_var is None:
                second_data_var = new_variable()
                data_var_to_const[second_data_var] = get_data_constants_for_tree_variable(can_var)[1]
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.input_predicate_to_unary_canonical_dict.inverse[can_predicate]
                data_conj.append((first_data_var, data_predicate, second_data_var))
            if not can_var.get_feature_list(): # If there's no RELEVANT binary predicate, just add the TOP one
                data_conj.append((first_data_var, self.TOP_PREDICATE, second_data_var))
            # Next, unfold children
            # We unfold directly the nodes, not the edges, because the presence of this node in the canonical encoding
            # (which is ensured by the lines above), already implies the presence of all these edges via colours 1,2,3
            for (_, col, _), child_var in can_var.children.items():
                bin_pred = internal_encoder.binary_pred_colour_dict.inverse[col]
                if bin_pred == self.col1:
                    # This is a binary node, and the edge is c1, so target must be unary node matching first var
                    unfold_variable_for_single(child_var, first_data_var)
                elif bin_pred == self.col2:
                    # Analogous to above
                    unfold_variable_for_single(child_var, second_data_var)
                    # Still a pair, but order must be reversed
                elif bin_pred == self.col3:
                    unfold_variable_for_pair(child_var, second_data_var, first_data_var)
                else:
                    raise ValueError(f"Binary fact in canonical atom uses predicate {bin_pred} which is not valid.")

        # Unfold unary canonical atom that unifies with a constant in the original signature (single).
        def unfold_variable_for_single(can_var, data_var: str):
            # First, add the relevant atoms.
            for feat in can_var.get_feature_list():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.input_predicate_to_unary_canonical_dict.inverse[can_predicate]
                data_conj.append((data_var, TYPE_PRED, data_predicate))
            # Next, unfold children
            # Here, for children via c1 and c2, we don't need to worry about unfolding the edge because it will be
            # created automatically by the existence of such children (TOP pred is added in each child if no features)
            # However, for children via c4, since the child is unary, we do need a TOP predicate, which unfolds the edge
            for (_, col, _), child_var in can_var.children.items():
                bin_pred = internal_encoder.binary_pred_colour_dict.inverse[col]
                if bin_pred == self.col1:
                    # This is a unary node, and the edge is c1, so target must be a binary node matching first var
                    unfold_variable_for_pair(child_var,first_data_var=data_var)
                elif bin_pred == self.col2:
                    # Analogous to above
                    unfold_variable_for_pair(child_var, second_data_var=data_var)
                elif bin_pred == self.col4:
                    # The target must be another unary node.
                    new_data_var = new_variable()
                    data_var_to_const[new_data_var] = get_data_constants_for_tree_variable(child_var)[0]
                    # A top fact must be added to unfold the edge connecting these two variables are connected.
                    data_conj.append((data_var, self.TOP_PREDICATE, new_data_var))
                    unfold_variable_for_single(child_var,new_data_var)
                else:
                    raise ValueError(f"Binary fact in canonical atom uses predicate {bin_pred} which is not valid.")

        if head_is_binary:
            unfold_variable_for_pair(can_conj.root_node, first_data_var=root_variables[0], second_data_var=root_variables[1])
        else:
            unfold_variable_for_single(can_conj.root_node, data_var=root_variables[0])

        data_conj = list(dict.fromkeys(data_conj)) # Remove potential duplicates

        # Process TOP predicate facts
        already_grounded_pairs = set() # Note variables that already appear together in a fact of the rule
        for s, p, o in data_conj:
            if p != TYPE_PRED and p != self.TOP_PREDICATE:
                already_grounded_pairs.add(frozenset((s, o)))
        # Now we filter out TOP_PREDICATE facts and replace them when necessary
        new_data_conj = data_conj.copy()
        for s, p, o in data_conj:
            if p == self.TOP_PREDICATE:
                new_data_conj.remove((s,p,o)) # Always remove from final conjunction
                if frozenset((s,o)) not in already_grounded_pairs:
                    a = data_var_to_const[s]
                    b = data_var_to_const[o]
                    t = self.term_for_pair((a, b)) # both this and the term for b a must exist
                    nz = torch.nonzero(cd_graph.features[cd_graph.node_names_to_indices[t]]).flatten()
                    if len(nz):
                        can_pred_idx = nz[0].item()
                        can_predicate = internal_encoder.unary_pred_position_dict.inverse[can_pred_idx]
                        new_data_conj.append((s,self.input_predicate_to_unary_canonical_dict.inverse[can_predicate],o))
                    else:
                        t = self.term_for_pair((b, a))
                        nz = torch.nonzero(cd_graph.features[cd_graph.node_names_to_indices[t]]).flatten()
                        assert len(nz) # if the feature of t-a-b was all 0, then that of t-b-a must have a 1
                        can_pred_idx = nz[0].item()
                        can_predicate = internal_encoder.unary_pred_position_dict.inverse[can_pred_idx]
                        new_data_conj.append((o,self.input_predicate_to_unary_canonical_dict.inverse[can_predicate],s))
                    already_grounded_pairs.add(frozenset((s,o)))

        return new_data_conj, root_variables

    # This is a rather specific function. Given two data variables y1 and y2, this returns a single variable if y1 y2
    # correspond to a single canonical variable y, and two variables if they correspond to a canonical variable each.
    def find_canonical_variable(self, can_variables_to_data_variables, y1, y2):
        binary = None
        unary_y1 = None
        unary_y2 = None
        for cvar in can_variables_to_data_variables:
            if len(can_variables_to_data_variables[cvar]) == 2 and (
                    (can_variables_to_data_variables[cvar][0] == y1 and can_variables_to_data_variables[cvar][1] == y2) or
                    (can_variables_to_data_variables[cvar][0] == y2 and can_variables_to_data_variables[cvar][1] == y1)):
                binary = cvar
            elif len(can_variables_to_data_variables[cvar]) == 1:
                if can_variables_to_data_variables[cvar][0] == y1:
                    unary_y1 = cvar
                elif can_variables_to_data_variables[cvar][0] == y2:
                    unary_y2 = cvar
        if binary:
            return [binary]
        elif unary_y1 and unary_y2:
            return [unary_y1, unary_y2]
        else:
            raise Exception("Error: data variables {} and {} do not seem to match any canonical variable. Bug.".format(y1,y2))
