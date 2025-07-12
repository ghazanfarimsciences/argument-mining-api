from abc import ABC, abstractmethod
from typing import List

from ..models.argument_units import LinkedArgumentUnits, UnlinkedArgumentUnits, ClaimPremiseRelationship

class ClaimPremiseLinker(ABC):
    """
    Abstract base class for linking claims to premises.
    This class defines the interface for claim-premise linking implementations.
    """

    @abstractmethod
    def link_claims_to_premises(self, unlinked_argument_units: UnlinkedArgumentUnits, max_retries: int) -> LinkedArgumentUnits:
        """
        Links claims to premises and returns the relationships.
        
        :param claims: List of claims to be linked.
        :param premises: List of premises to be linked.
        :return: A list of relationships between claims and premises.
        """
        pass