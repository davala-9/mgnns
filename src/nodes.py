#  A map from every constant to the corresponding node, and back.

global const_node_dict
const_node_dict = {}

global node_const_dict
node_const_dict = {}


def add_node_for_constant(constt):
    assert constt not in const_node_dict
    current_length = len(const_node_dict)
    const_node_dict[constt] = current_length + 1
    node_const_dict[current_length + 1] = constt
    return


def get_node_for_constant(constt):
    if constt not in const_node_dict:
        add_node_for_constant(constt)
    return const_node_dict[constt]
