from fact_explanation import FactExplainer
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from src.utils.bitset import BitSet
from src.model.cd_graph import TraceCollector
from src.model.gnn_transformation import apply_model

# OPTIMISATION 1
# Make a dictionary of all unary atoms of gamma_i and compute an ``influence'' value of each
# by creating the minimal sub-rule of Gamma_i that connects this atom with the head variable.
# Ground such rule and apply the GNN to it, then check the value of the node and position corresponding to the head
# fact. If zero, start travelling backwards through the relevant part of the computation graph; at each node of the
# computation tree, sum the values of the relevant positions of the activations. Keep doing this until not zero.

# Compute the influence value for this contributor
def compute_influence(temp_tree: TreeShapedConjunction, fe: FactExplainer, trace: TraceCollector, variable: Variable,
                      new_vars_to_original_vars: dict[Variable,Variable], pos: int,
                      var_layer_mask: dict[tuple[Variable, int], BitSet],
                      contributors_to_influence_dict: dict[(Variable,int),(int, float)]):
    contribution_score = 0
    current_variable = temp_tree.root_node
    next_variable = next(iter(current_variable.children.values()), None)
    for l in range(fe.model.num_layers, -1, -1):
        # Identify the correct variable for this step of the computation.
        # This matters because there might be level jumps between variables.
        if 0 < l == next_variable.level:
            current_variable = next_variable
            next_variable = next(iter(current_variable.children.values()), None)
        original_variable = new_vars_to_original_vars[current_variable]
        for k in var_layer_mask[original_variable,l].elements():
            current_variable_id = temp_tree.index_tree.node_to_id[current_variable]
            contribution_score += trace.activations[l][current_variable_id][k]
        if contribution_score > 0:
            contributors_to_influence_dict[(variable, pos)] = (l, contribution_score)
            break
    contributors_to_influence_dict[(variable, pos)] = (0,0)


def apply_optimisation(fe: FactExplainer, rule_body: TreeShapedConjunction, predicate_position: int,
                         var_const_idx: dict[Variable, int], var_layer_mask: dict[tuple[Variable, int], BitSet]):

    # First we compute a dictionary that maps each unary atom U_j(y) in \Gamma_i, represented by the pair (y,j), to a
    # "contribution": a pair (layer, value) representing the first layer (from the end) where this has a non-zero
    # contribution, and "value" is the value of this contribution.
    contributors_to_influence_dict = {}
    for variable in var_const_idx:  # Iterate over each unary fact in the input rule
        if variable.features.is_empty():
            continue
        temp_tree, new_vars_to_original_vars = rule_body.get_subtree_for(variable)
        new_variable = new_vars_to_original_vars.inverse[variable]
        relevant_positions = var_layer_mask[(variable,0)].elements()
        for pos in relevant_positions:
            new_variable.features = BitSet.from_subset(variable.features.dimension,{pos}) # mask holds only relevant pos
            temp_cd_graph = temp_tree.as_cd_graph() # recall that this ensures that node id matches the tree's index
            trace = TraceCollector()
            apply_model(temp_cd_graph, fe.device, fe.model,trace)
            compute_influence(temp_tree,fe,trace,variable,new_vars_to_original_vars,pos,var_layer_mask,
                              contributors_to_influence_dict)
    # Next we sort contributors by contribution.
    contributors_list = []
    for (y, j), (s,l) in contributors_to_influence_dict.items():
        contributors_list.append((s, l, y, j))
        contributors_list = sorted(contributors_list)

    # Start trying rules from the empty rule, adding atoms in increasing order of contribution
   # Initialise empty rule
    new_root = rule_body.root_node.shallow_copy()
    temp_tree = TreeShapedConjunction(root_node=new_root, n_colours=rule_body.n_colours)
    while contributors_list:
        influence_score, layer, var, pos = contributors_list.pop()
        new_tree = rule_body.get_subtree_for(var)
        temp_tree.append(new_tree)
        temp_cd_graph = temp_tree.as_cd_graph()
        output_graph = apply_model(temp_cd_graph, fe.device, fe.model)
        if output_graph.features[0][predicate_position] > fe.threshold:
            return temp_tree
    raise AssertionError("This part of the code should not be reachable. There's a bug in Optimisation 1")

    # TODO ITP: maybe a clean-up step like in approximation 2 can help make improvements