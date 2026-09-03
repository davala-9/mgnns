from src.encodings.canonical import CanonicalEncoderDecoder
from src.model.cd_graph import CDGraph
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from abc import ABC, abstractmethod
from dataclasses import dataclass

ineq_pred = "owl:differentFrom"

# This is an auxiliary class for the "unfold" method. It helps unfold a rule that can be grounded on a specific fact.
@dataclass
class GroundContext:
    fact: [str,str,str] # Fact to be explained
    graph: CDGraph # Graph on which the fact was predicted
    canonical_variable_to_constant_index: dict[int,int] # map of pre-unfold variables to constants in the cd-graph

class NonCanonicalEncoder(ABC):

    canonical_unary_predicates = list
    canonical_binary_predicates = list

    @abstractmethod
    def encode_dataset(self, dataset: set[tuple], **kwargs) -> set[tuple]:
        pass

    @abstractmethod
    def decode_dataset(self, dataset: set[tuple]) -> set[tuple]:
        pass


    # Maps a canonical unary fact to the corresponding unary or binary data fact.
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
    # returns all possible unfoldings, and the rule head.
    @abstractmethod
    def unfold_all(self, can_conj:TreeShapedConjunction, internal_encoder:CanonicalEncoderDecoder, head_predicate: str):
        pass

    # This function takes a tree-shaped conjunction expressed in the Canonical Signature and a GroundContext, and
    # then returns ONE specific unfolding that can be grounded in the data-signature dataset, and the rule head.
    @abstractmethod
    def unfold_match_ground(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder,
               head_predicate: str, grounding_context: GroundContext):
        pass


