from dataclasses import dataclass
from typing import List, Dict
from uuid import UUID as uuid


@dataclass
class ArgumentUnit:
    """Represents an identified argument unit (ADU)"""
    uuid: uuid
    text: str
    start_pos: int
    end_pos: int
    type: str  # 'claim' or 'premise'
    confidence: float

@dataclass
class StanceRelation:
    """Links one premise to one claim with a stance label and confidence"""
    claim_id: uuid
    premise_id: uuid
    stance: str  # 'pro' or 'con'
    confidence: float

# This class is used to represent unlinked argument units (ADUs) before they are linked by stance relations.
# This will be done via the LLM (ChatGPT) in the next step. 
# That has to be done before we can do the stance classification.
@dataclass
class UnlinkedArgumentUnits:
    """Represents unlinked argument units (ADUs)"""
    claims: List[ArgumentUnit]
    premises: List[ArgumentUnit]



@dataclass
class ClaimPremiseRelationship:
    """Represents a relationship between a claim and premise"""
    claim_id: uuid
    premise_ids: List[uuid]

# This is the result of the linking step, where we linked claims to premises.
# 
@dataclass
class LinkedArgumentUnits:
    """Represents linked argument units with their relationships"""
    # Mapping of claim UUIDs to their corresponding premise UUIDs
    claims : List[ArgumentUnit]
    premises: List[ArgumentUnit]
    claims_premises_relationships: List[ClaimPremiseRelationship]


# We call Bert classify stance with the LinkedArgumentUnits + List of Claims and Premises

# This is the Output class for the entire Linked Argument Structure, 
# here you can use the Ids in the Stance Relations to link the claims and premises
# This class will be used to create the final Graph Representation.
@dataclass
class LinkedArgumentUnitsWithStance:
    """
    Represents linked argument units with stance relations.
    The stance relations are used to link claims to premises with their stance used for the graph representation.
    """
    original_text: str
    claims: List[ArgumentUnit]
    premises: List[ArgumentUnit]
    stance_relations: List[StanceRelation]
