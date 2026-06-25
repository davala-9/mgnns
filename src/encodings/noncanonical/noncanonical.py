from src.encodings.canonical import CanonicalEncoderDecoder
from src.rule_extraction.tree_shaped_conjunction import TreeShapedConjunction
from abc import ABC, abstractmethod

ineq_pred = "owl:differentFrom"

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
    def unfold(self, can_conj: TreeShapedConjunction, internal_encoder: CanonicalEncoderDecoder, **kwargs):
        pass




