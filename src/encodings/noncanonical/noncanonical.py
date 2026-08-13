from src.encodings.canonical import CanonicalEncoderDecoder
from src.model.cd_graph import CDGraph
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction, Variable
from abc import ABC, abstractmethod
from dataclasses import dataclass

ineq_pred = "owl:differentFrom"

# This is an auxiliary class for the "unfold" method of non-canonical decodings.
# It helps unfold a rule whenever we want an unfolding that can be grounded on a specific fact.
@dataclass
class GroundContext:
    fact: [str,str,str]
    graph: CDGraph
    canonical_variable_to_constant_index: dict[Variable,int]

class NonCanonicalEncoder(ABC):

    canonical_unary_predicates = list
    canonical_binary_predicates = list

    @abstractmethod
    def encode_dataset(self, dataset: set[tuple], **kwargs) -> set[tuple]:
        pass

    @abstractmethod
    def decode_dataset(self, dataset: set[tuple]) -> set[tuple]:
        pass

    @abstractmethod
    def decode_fact(self, s: str, p:str, o:str) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def get_canonical_equivalent(self, fact: tuple[str, str, str]) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def unary_can_predicate_to_data_predicate(self, predicate:str):
        pass

    @abstractmethod
    def unary_can_predicate_to_data_predicate_arity(self, predicate:str):
        pass

    # This function takes a tree-shaped conjunction expressed in the Canonical Signature and
    # returns all possible unfoldings, together with the rule head.
    # (Returning both together incurs some coupling, but it allows us to optimise slightly the extraction procedure)
    @abstractmethod
    def unfold_all(self, can_conj:TreeShapedConjunction, internal_encoder:CanonicalEncoderDecoder, head_predicate: str):
        pass

    # This function takes a tree-shaped conjunction expressed in the Canonical Signature and some extra information
    # about how this conjunction is grounded in the canonical dataset, and then returns ONE specific unfolding that
    # can be grounded in the data dataset, together with the rule head.
    @abstractmethod
    def unfold_match_ground(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder,
               head_predicate: str, grounding_context: GroundContext):
        pass


