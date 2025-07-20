import openai
import re 

from uuid import UUID, uuid4
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship
from typing import List
from ..config import OPENAI_KEY
from ...log import log


#TODO: Implement Pydantic
#NOTE: classify_stance is not implement yet 


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())
      
class OpenAILLMClassifier (AduAndStanceClassifier): 
    def __init__ (self): 
        self.client = openai.OpenAI(api_key=OPENAI_KEY)
        self.system_prompt_adu_classification = """You are an assistant for argument mining.
                        Classify the following sentence as either a 'claim' or a 'premise'.
                        Only return one word: claim or premise.
                        """
        self.system_prompt_stance_classification = """You are an assistant for argument mining.
                        You are given a claim and evidence (premise).
                        Determine whether the evidence supports ('pro') or refutes ('con') the claim.
                        Respond only with one word: "pro" or "con".
                        """
        
    def classify_sentence (self, sentence: str, model: str = "gpt-4-turbo") -> str: 
        """
        Executes the prompt to classify a single sentence as a 'claim' or 'premise'.
        Includes a retry mechanism with fallback to gpt-3.5-turbo.
        """
        user_prompt = f'Sentence: "{sentence}"\nWhat is it?'
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt_adu_classification.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            log().warning(f"❌ Model {model} failed for ADU classification: {e}")
            if model != "gpt-3.5-turbo":
                log().info("Attempting with gpt-3.5-turbo for ADU classification.")
                return self.classify_sentence(sentence, model="gpt-3.5-turbo")
            log().error("Failed ADU classification after retries. Defaulting to 'premise'.")
            return "premise" # Fallback if all models fail
        
    def classify_stance_single(self, claim_text: str, premise_text: str, model: str = "gpt-4-turbo") -> str:
        """
        Classifies the stance between a single claim and premise.
        Includes a retry mechanism with fallback to gpt-3.5-turbo.
        """
        user_prompt = f"""Claim: {claim_text}
Evidence: {premise_text}
Stance:"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt_stance_classification.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.2,
                max_tokens=5 # Keep it short for 'pro' or 'con'
            )
            result = response.choices[0].message.content.strip().lower()
            if any(x in result for x in ("refute", "con")):
                return "con"
            elif any(x in result for x in ("support", "pro")):
                return "pro"
            else:
                log().warning(f"Unexpected stance output: {result} for Claim: '{claim_text}' | Premise: '{premise_text}'")
                return "unidentified"
        except Exception as e:
            log().warning(f"❌ Model {model} failed for stance classification: {e}")
            if model != "gpt-3.5-turbo":
                log().info("Attempting with gpt-3.5-turbo for stance classification.")
                return self.classify_stance_single(claim_text, premise_text, model="gpt-3.5-turbo")
            log().error("Failed stance classification after retries. Defaulting to 'unidentified'.")
            return "unidentified" # Fallback if all models fail

            
    def classify_adus(self, text: str) -> List[ArgumentUnit]:
        """
        Extracts and labels argumentative units from text.
        Args:
            text (str): The raw input document.  
        Returns:
            List[ArgumentUnit]: Each unit is a Claim or Premise with
                                 start/end character positions and confidence.
        """
        # Step 1: Split paragraph into sentences 
        sentences = split_into_sentences(text)
        log().info(f"Found {len(sentences)} sentences in the input text")

        # Step 2: Predict ADU Type for every sentence 
        argument_mining_list = []
        model_to_use = "gpt-4-turbo" # Can be configured
        current_pos = 0 
        for sentence in sentences: 
            adu_type = self.classify_sentence(sentence, model_to_use)
            
            # Find start and end positions. This is a simplification and might need
            # more robust handling for complex texts (e.g., if sentence appears multiple times)
            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence) if start_pos != -1 else current_pos
            
            log().debug(f"Sentence: '{sentence}' | Predicted as: {adu_type} | Start: {start_pos} | End: {end_pos}")
            
            adu = ArgumentUnit(uuid=uuid4(), text=sentence, type=adu_type, start_pos=start_pos, end_pos=end_pos, confidence=1.0) # OpenAI doesn't directly provide confidence score in this API call
            argument_mining_list.append(adu)
            
            if start_pos != -1: # Update current position for next search
                current_pos = end_pos + 1 # +1 to skip the space/punctuation
            
        return argument_mining_list
    
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.

        :param linked_argument_units: The structured links between claims and premises.
        :param originalText: The original text from which the claims and premises were extracted (for context only).
        :return: LinkedArgumentUnitsWithStance object representing the final stance graph.
        """
        result_linked_arguments: List[StanceRelation] = []
        model_to_use = "gpt-4-turbo" # Can be configured

        for relation in linked_argument_units.claims_premises_relationships:
            # Find the claim object
            claim = next((c for c in linked_argument_units.claims if c.uuid == relation.claim_id), None)
            if claim is None:
                log().warning(f"No Claim found for this relationship: {relation} --> Continuing loop")
                continue

            log().debug(f"Processing Claim: {claim.text}")

            if not relation.premise_ids:
                log().warning(f"No premises found for claim '{claim.text}' ---> Continuing loop")
                continue

            for pid in relation.premise_ids:
                premise = next((p for p in linked_argument_units.premises if p.uuid == pid), None)
                if not premise:
                    log().warning(f"No premise found for the ID {pid} ---> Continuing loop")
                    continue

                log().debug(f"  → With Premise: {premise.text}")
                
                stance_relationship = self.classify_stance_single(claim.text, premise.text, model_to_use)
                
                log().debug(f"Claim: '{claim.text}' | Premise: '{premise.text}' -> Relationship: {stance_relationship}")
                
                result_linked_arguments.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=stance_relationship,
                        confidence=1.0 # OpenAI doesn't directly provide confidence score in this API call
                    )
                )

        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=result_linked_arguments
        )

def test_model(): 
    log().info("Running OpenAI LLM Classifier test...")
    # Create UUIDs
    claim1_id = uuid4()
    claim2_id = uuid4()
    premise1_id = uuid4()
    premise2_id = uuid4()
    premise3_id = uuid4()
    
    # Claims
    claims = [
        ArgumentUnit(uuid=claim1_id, text="Climate change is a serious threat.", start_pos=0, end_pos=34, type="claim", confidence=0.98),
        ArgumentUnit(uuid=claim2_id, text="Education improves social mobility.", start_pos=35, end_pos=70, type="claim", confidence=0.95)
    ]
    
    # Premises
    premises = [
        ArgumentUnit(uuid=premise1_id, text="Rising temperatures are affecting ecosystems.", start_pos=71, end_pos=120, type="premise", confidence=0.97),
        ArgumentUnit(uuid=premise2_id, text="CO2 levels have increased drastically.", start_pos=121, end_pos=160, type="premise", confidence=0.96),
        ArgumentUnit(uuid=premise3_id, text="Access to schooling enables better job opportunities.", start_pos=161, end_pos=210, type="premise", confidence=0.96)
    ]
    
    # Relationships
    relationships = [
        ClaimPremiseRelationship(claim_id=claim1_id, premise_ids=[premise1_id, premise2_id]),
        ClaimPremiseRelationship(claim_id=claim2_id, premise_ids=[premise3_id])
    ]
    
    # LinkedArgumentUnits test object
    linked_argument_units_test = LinkedArgumentUnits(
        claims=claims,
        premises=premises,
        claims_premises_relationships=relationships
    )
    
    original_text = "Climate change is a serious threat. Education improves social mobility. Rising temperatures are affecting ecosystems. CO2 levels have increased drastically. Access to schooling enables better job opportunities."
    
    # Test ADU classification
    log().info("Testing classify_adus method:")
    openai_classifier = OpenAILLMClassifier()
    identified_adus = openai_classifier.classify_adus(original_text)
    for adu in identified_adus:
        log().info(f"Identified ADU: Type={adu.type}, Text='{adu.text}'")

    # Test Stance classification
    log().info("\nTesting classify_stance method:")
    result_stance = openai_classifier.classify_stance(linked_argument_units_test, original_text)
    
    log().debug(result_stance)

    for relation in result_stance.stance_relations:
        claim_text = next(c.text for c in result_stance.claims if c.uuid == relation.claim_id)
        premise_text = next(p.text for p in result_stance.premises if p.uuid == relation.premise_id)
        log().info(f"Stance: Claim: '{claim_text}' | Premise: '{premise_text}' | Stance: {relation.stance}")


if __name__ == "__main__":
    test_model()