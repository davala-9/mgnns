import torch

from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.noncanonical import NonCanonicalEncoder, GroundContext
from bidict import bidict

from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from src.utils.utils import TYPE_PRED

class ICLREncoderDecoder(NonCanonicalEncoder):

    # Fresh predicates that correspond to colours c1, c2, c3, c4 in the paper. Abbreviations match paper names
    col1, col2, col3, col4 = "binary-pred-1", "binary-pred-2", "binary-pred-3", "binary-pred-4"

    # This is a placeholder predicate used for unfoldings. It simply says "these two appear together in the dataset"
    TOP_PREDICATE = "top-pred"

    def __init__(self, load_from_document=None, unary_predicates=None, binary_predicates=None):
        self.canonical_binary_predicates = [self.col1, self.col2, self.col3, self.col4]
        self.data_pred_to_unary_canonical = bidict()
        self.data_pred_to_arity = {}
        if load_from_document is not None:
            self.data_unary_predicates = []
            self.data_binary_predicates = []
            for line in open(load_from_document, 'r').readlines():
                input_predicate, canonical_predicate, arity = line.split()
                self.data_pred_to_unary_canonical[input_predicate] = canonical_predicate
                self.data_pred_to_arity[input_predicate] = int(arity)
                if int(arity)==1:
                    self.data_unary_predicates.append(input_predicate)
                else:
                    self.data_binary_predicates.append(input_predicate)
        else:
            assert set(unary_predicates).isdisjoint(set(binary_predicates)) # Sanity check
            for pred in unary_predicates:
                self.data_pred_to_unary_canonical[pred] = pred
                self.data_pred_to_arity[pred] = 1
            for pred in binary_predicates:
                self.data_pred_to_unary_canonical[pred] = "unary-for-{}".format(pred)
                self.data_pred_to_arity[pred] = 2
            self.data_unary_predicates = unary_predicates
            self.data_binary_predicates = binary_predicates
        # Maps pairs of constants to a new single term
        self.pair_term_dict = bidict()
        self.canonical_unary_predicates = list(self.data_pred_to_unary_canonical.inverse.keys())

    # Save format (predicate \t new_predicate \t arity)
    def save_to_file(self, target_file):
        output = open(target_file, 'w')
        for input_predicate in self.data_pred_to_unary_canonical:
            output.write("{}\t{}\t{}\n".format(input_predicate,
                                               self.data_pred_to_unary_canonical[input_predicate],
                                               self.data_pred_to_arity[input_predicate]))
        output.close()

    def term_for_pair(self, pair):
        if pair not in self.pair_term_dict:
            self.pair_term_dict[pair] = "term-for-{}-{}".format(pair[0], pair[1])
        return self.pair_term_dict[pair]

    def encode_fact(self, fact):
        encoded_dataset = []
        s, p, o = fact
        if p == TYPE_PRED:
            encoded_dataset.append((s, p, self.data_pred_to_unary_canonical[o]))
        else:
            a = s    # We rename to the paper's notation to make the code easier to read and write
            b = o
            ab = self.term_for_pair((a, b))
            ba = self.term_for_pair((b, a))
            encoded_dataset.append((ab, TYPE_PRED, self.data_pred_to_unary_canonical[p]))
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
                if not self.is_can_const_for_pair(s):
                    constants.add(s)
                if p != TYPE_PRED and not self.is_can_const_for_pair(o):
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
            return s, p, self.data_pred_to_unary_canonical[o]
        else:
            ab = self.term_for_pair((s, o))
            return ab, TYPE_PRED, self.data_pred_to_unary_canonical[p]

    def unary_can_predicate_to_data_predicate(self, predicate:str):
        return self.data_pred_to_unary_canonical.inverse[predicate]

    def unary_can_predicate_to_data_predicate_arity(self, predicate: str):
        return self.data_pred_to_arity[
            self.data_pred_to_unary_canonical.inverse[predicate]]

    def decode_dataset(self, canonical_dataset):
        return {decoded for s, p, o in canonical_dataset
                if (decoded := self.decode_fact(s, p, o)) is not None}

    def decode_fact(self, s, p, o):
        assert(p == TYPE_PRED) # ICLR decodes only unary canonical facts. Binary can facts have no data equivalent.
        data_predicate = self.data_pred_to_unary_canonical.inverse[o]
        if self.is_can_const_for_pair(s) and self.data_pred_to_arity[data_predicate] == 2:
            a, b = self.pair_term_dict.inverse[s]
            return a, self.data_pred_to_unary_canonical.inverse[o], b
        elif not self.is_can_const_for_pair(s) and self.data_pred_to_arity[data_predicate] == 1:
            return s, TYPE_PRED, self.data_pred_to_unary_canonical.inverse[o]
        return None

    def is_can_const_for_pair(self, const):
        return const in self.pair_term_dict.inverse

    # We traverse the canonical conjunction, unfolding as we go.
    # We use a slight optimisation: we KNOW whether the head variable represents a constant pair or a single
    # constant in the input data, based on the arity of the head predicate. Because of this, we can propagate this
    # information to know, for each CANONICAL variable, whether it represents also a constant pair or a single
    # constant in the input data. This halves the size of the output program.
    # We unfold mainly canonic unary atoms, which turn into either unary or binary data atoms.
    # Canonical binary atoms must sometimes be unfolded too.
    def unfold_all(self,can_conj: TreeShapedConjunction,internal_encoder: CanonicalEncoderDecoder,head_predicate: str):
        # This has multiple uses in the unfolding
        head_predicate_arity = self.data_pred_to_arity[head_predicate]
        head_is_binary = head_predicate_arity == 2

        # Variable manager
        data_var_prefix = "X"  # Variables in the unfolded conjunction are of the form Xn, for n a number
        data_var_counter = 0
        def new_variable():  # Aux method to create new variables
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)

        # Define root variables
        root_variables = [data_var_prefix + str(data_var_counter)]  # X0 is always a root variable
        if head_is_binary:
            second_root_data_var = new_variable()
            root_variables.append(second_root_data_var)

        # Aux method, takes two variables and returns all possible binary atoms that use them both
        def all_binary_atoms(data_var_1, data_var_2):
            all_atoms = set([])
            equal_vars = data_var_1 == data_var_2
            for binary_predicate in self.data_binary_predicates:
                all_atoms.add((data_var_1, binary_predicate, data_var_2))
                if not equal_vars:
                    all_atoms.add((data_var_2, binary_predicate, data_var_1))
            return all_atoms

        # Unfold variable that we know represents a pair of data_constants.
        def unfold_variable_for_pair(data_conj, var_id: int, first_data_var: str = None, second_data_var: str = None):
            # Create any missing variables
            assert first_data_var is not None or second_data_var is not None  # We should know at least one of them.
            if first_data_var is None:
                first_data_var = new_variable()
            if second_data_var is None:
                second_data_var = new_variable()

            # Extend conjunction with all relevant unary atoms. Can branch into multiple options.
            new_data_conjs = []
            if can_conj.features[var_id].elements():
                for feat in can_conj.features[var_id].elements():
                    can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                    data_predicate = self.data_pred_to_unary_canonical.inverse[can_predicate]
                    data_conj.append((first_data_var, data_predicate, second_data_var))
                new_data_conjs.append(data_conj)
            else:  # If there's no RELEVANT binary predicate, we must add all binary predicates as per the encoding
                for atom in all_binary_atoms(first_data_var,second_data_var):
                    new_data_conjs.append(data_conj + [atom])

            # Extend all possible conjunctions in the previous step recursively, by unfolding the children of can_var
            # We unfold directly the nodes, not the edges, because the presence of this node in the canonical encoding
            # (which is ensured by the lines above), already implies the presence of all these edges via colours 1,2,3
            # For each colour, we consider all possible ways to expand the current conjunctions
            for (_, col, _), child_var in can_conj.children[var_id].items():
                bin_pred = internal_encoder.binary_pred_colour_dict.inverse[col]
                next_new_data_conjs = []
                for new_data_conj in new_data_conjs:
                    if bin_pred == self.col1:
                        # This is a binary node, and the edge is c1, so target must be unary node matching first var
                        next_new_data_conjs.extend(
                            unfold_variable_for_single(new_data_conj, child_var, first_data_var))
                    elif bin_pred == self.col2: # Analogous to above
                        next_new_data_conjs.extend(
                                unfold_variable_for_single(new_data_conj, child_var, second_data_var))
                    elif bin_pred == self.col3: # Still a pair, but order must be reversed
                        next_new_data_conjs.extend(
                            unfold_variable_for_pair(new_data_conj, child_var, second_data_var, first_data_var))
                    else:
                        continue # if using self.col4, this rule will never match a dataset and should be discarded
                new_data_conjs = next_new_data_conjs

            return new_data_conjs

        # Unfold variable that we know represents a single constant
        def unfold_variable_for_single(data_conj, var_id: int, data_var: str):
            # First, extend conjunction with the relevant unary atoms.
            for feat in can_conj.features[var_id].elements():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.data_pred_to_unary_canonical.inverse[can_predicate]
                data_conj.append((data_var, TYPE_PRED, data_predicate))

            # Next, unfold children
            # Here, for children via c1 and c2, we don't need to worry about unfolding the edge because such edge will
            # always be created by the encoder due to the existence of each such children, which is a pair
            # However, for children via c4, since the child is unary, we simply add all binary predicates as per encoder
            new_data_conjs = [data_conj]
            for (_, col, _), child_var in can_conj.children[var_id].items():
                bin_pred = internal_encoder.binary_pred_colour_dict.inverse[col]
                next_new_data_conjs = []
                for new_data_conj in new_data_conjs:
                    if bin_pred == self.col1:
                        # This is a unary node, and the edge is c1, so target must be a binary node matching first var
                        next_new_data_conjs.extend(
                            unfold_variable_for_pair(new_data_conj, child_var, first_data_var=data_var))
                    elif bin_pred == self.col2: # Analogous to above
                        next_new_data_conjs.extend(
                            unfold_variable_for_pair(new_data_conj, child_var, second_data_var=data_var))
                    elif bin_pred == self.col4: # The target must be another unary node.
                        new_data_var = new_variable()
                        # We must consider an extension with each possible binary predicate
                        for atom in all_binary_atoms(data_var,new_data_var):
                            next_new_data_conjs.extend(
                                unfold_variable_for_single(new_data_conj + [atom],child_var, new_data_var))
                    else:
                        continue # if this rule uses self.col3 it will never match a dataset and should be discarded
                new_data_conjs = next_new_data_conjs
            return new_data_conjs

        # Unfolding start
        if head_is_binary:
            data_conjs = unfold_variable_for_pair(data_conj=[], var_id=0, first_data_var=root_variables[0],
                                                  second_data_var=root_variables[1])
        else:
            data_conjs = unfold_variable_for_single(data_conj=[], var_id=0,data_var=root_variables[0])
        data_conjs = [list(dict.fromkeys(dc)) for dc in data_conjs]  # Remove potential duplicates

        if head_is_binary:
            head = (root_variables[0], head_predicate, root_variables[1])
        else:
            head = (root_variables[0], TYPE_PRED, head_predicate)

        return data_conjs, head


    # We traverse the canonical conjunction, unfolding as we go.
    # We unfold mainly canonic unary atoms, which turn into either unary or binary data atoms.
    # Canonical binary atoms are often superfluous, but in some cases require the addition of a TOP predicate fact
    # Note that head_predicate is the *data* predicate
    def unfold_match_ground(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder,
                            head_predicate: str, grounding_context: GroundContext):

        data_conj = [] # The output conjunction. List, not a TreeLikeConj, because unfolding might break the tree struct

        # This has multiple uses in the unfolding
        head_predicate_arity = self.data_pred_to_arity[head_predicate]
        head_is_binary = head_predicate_arity == 2

        # Map from canonical variables to the indices of the graph nodes they represent
        can_var_to_can_const_idx = grounding_context.canonical_variable_to_constant_index
        # Map from data variables to the data constants they are grounded to
        data_var_to_data_const = {}

        # A canonical variable is grounded into a canonical constant, which in turn matches one or two data variables.
        def can_variable_to_can_constant(var_id: int):
            return grounding_context.graph.node_names[can_var_to_can_const_idx[var_id]]

        # Returns the one or two data variables as a list.
        def get_data_constants_for_can_variable(var_id: int):
            canonical_constant = can_variable_to_can_constant(var_id)
            if canonical_constant in self.pair_term_dict.inverse:
                return list(self.pair_term_dict.inverse[canonical_constant])
            else:
                return [canonical_constant]

        # Variable manager
        data_var_prefix = "X" # Variables in the unfolded conjunction are of the form Xn, for n a number
        data_var_counter = 0
        def new_variable(): # Aux method to create new variables
            nonlocal data_var_counter
            data_var_counter += 1
            return data_var_prefix + str(data_var_counter)
        root_variables = [data_var_prefix + str(data_var_counter)] # X0 is always a root variable

        data_var_to_data_const[root_variables[0]] = get_data_constants_for_can_variable(0)[0]

        if head_is_binary:
            second_root_data_var = new_variable()
            root_variables.append(second_root_data_var)
            data_var_to_data_const[root_variables[1]] = get_data_constants_for_can_variable(0)[1]

        # Unfold unary canonical atom that unifies with a canonical constant for a pair of data_constants.
        def unfold_variable_for_pair(var_id: int, first_data_var: str=None, second_data_var: str=None):
            # First, add the relevant atoms.
            assert first_data_var is not None or second_data_var is not None # We should know at least one of them.
            if first_data_var is None:
                first_data_var = new_variable()
                data_var_to_data_const[first_data_var] = get_data_constants_for_can_variable(var_id)[0]
            if second_data_var is None:
                second_data_var = new_variable()
                data_var_to_data_const[second_data_var] = get_data_constants_for_can_variable(var_id)[1]
            for feat in can_conj.features[var_id].elements():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.data_pred_to_unary_canonical.inverse[can_predicate]
                data_conj.append((first_data_var, data_predicate, second_data_var))
            if not can_conj.features[var_id].elements(): # If there's no RELEVANT binary predicate, just add the TOP one
                data_conj.append((first_data_var, self.TOP_PREDICATE, second_data_var))
            # Next, unfold children
            # We unfold directly the nodes, not the edges, because the presence of this node in the canonical encoding
            # (which is ensured by the lines above), already implies the presence of all these edges via colours 1,2,3
            for (_, col, _), child_var in can_conj.children[var_id].items():
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
        def unfold_variable_for_single(var_id, data_var: str):
            # First, add the relevant atoms.
            for feat in can_conj.features[var_id].elements():
                can_predicate = internal_encoder.unary_pred_position_dict.inverse[feat]
                data_predicate = self.data_pred_to_unary_canonical.inverse[can_predicate]
                data_conj.append((data_var, TYPE_PRED, data_predicate))
            # Next, unfold children
            # Here, for children via c1 and c2, we don't need to worry about unfolding the edge because it will be
            # created automatically by the existence of such children (TOP pred is added in each child if no features)
            # However, for children via c4, since the child is unary, we do need a TOP predicate, which unfolds the edge
            for (_, col, _), child_var in can_conj.children[var_id].items():
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
                    data_var_to_data_const[new_data_var] = get_data_constants_for_can_variable(child_var)[0]
                    # A top fact must be added to unfold the edge connecting these two variables are connected.
                    data_conj.append((data_var, self.TOP_PREDICATE, new_data_var))
                    unfold_variable_for_single(child_var,new_data_var)
                else:
                    raise ValueError(f"Binary fact in canonical atom uses predicate {bin_pred} which is not valid.")

        if head_is_binary:
            unfold_variable_for_pair(0, first_data_var=root_variables[0], second_data_var=root_variables[1])
        else:
            unfold_variable_for_single(0, data_var=root_variables[0])

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
                    a = data_var_to_data_const[s]
                    b = data_var_to_data_const[o]
                    assert (a,b) in self.pair_term_dict # both this and the term for b a must exist
                    t = self.term_for_pair((a, b))
                    nz = torch.nonzero(
                        grounding_context.graph.features[grounding_context.graph.node_names_to_indices[t]]).flatten()
                    if len(nz):
                        can_pred_idx = nz[0].item()
                        can_predicate = internal_encoder.unary_pred_position_dict.inverse[can_pred_idx]
                        new_data_conj.append((s, self.data_pred_to_unary_canonical.inverse[can_predicate], o))
                    else:
                        assert (b, a) in self.pair_term_dict
                        t = self.term_for_pair((b, a))
                        nz = torch.nonzero(
                            grounding_context.graph.features[grounding_context.graph.node_names_to_indices[t]]).flatten()
                        assert len(nz) # if the feature of t-a-b was all 0, then that of t-b-a must have a 1
                        can_pred_idx = nz[0].item()
                        can_predicate = internal_encoder.unary_pred_position_dict.inverse[can_pred_idx]
                        new_data_conj.append((o, self.data_pred_to_unary_canonical.inverse[can_predicate], s))
                    already_grounded_pairs.add(frozenset((s,o)))

        if head_is_binary:
            head = (root_variables[0], head_predicate, root_variables[1])
        else:
            head = (root_variables[0], TYPE_PRED, head_predicate)

        return new_data_conj, head