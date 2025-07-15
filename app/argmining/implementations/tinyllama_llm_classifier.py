import torch
import re 

from transformers import AutoTokenizer, AutoModelForCausalLM
from uuid import uuid4
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship
from typing import List
from ..log import log
from ..config import HF_TOKEN 

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

class TinyLLamaLLMClassifier (AduAndStanceClassifier): 
    def __init__ (self): 
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_auth_token=HF_TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, use_auth_token=HF_TOKEN)
    
    def run_prompt(self, prompt, max_new_tokens=10, max_retries=3, retry_delay=1):
        """
        Runs the prompt on the model with a retry mechanism.
        Args:
            prompt (str): The prompt to send to the model.
            mode (str): The mode for the prompt (not used internally).
            max_new_tokens (int): Number of new tokens to generate.
            max_retries (int): Maximum number of retry attempts.
            retry_delay (int or float): Delay (in seconds) between retries.
        Returns:
            str: The model's response.
        Raises:
            Exception: If all attempts fail.
        """
        import time
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return result[len(prompt):].strip()
            except Exception as e:
                log().warning(f"⚠️ LLM failed on attempt {attempt}/{max_retries}: {e}")
                last_exception = e
                if attempt < max_retries:
                    time.sleep(retry_delay)
        # If we reach here, all attempts failed
        raise Exception(f"Model failed to run the prompt after {max_retries} attempts") from last_exception
        
        
    def classify_adus(self, text: str) -> List[ArgumentUnit]:
        """Extracts and labels argumentative units from text.    
        Args:
            text (str):  The raw input document.    
        Returns:
            List[ArgumentUnit]:  Each unit is a Claim or Premise
        """
        # Step1: Split paragraph into sentences 
        sentences = split_into_sentences(text)
        log().info(f"Found {len(sentences)} in the input text")
        # Step 2: Iterate through sentences and classify them
        argument_mining_list = []
        for sentence in sentences : 
            prompt = f"Classify the following sentence as either a 'claim' or a 'premise':\n\"{sentence}\"\nAnswer:"

            response = self.run_prompt(prompt)
            adu_type = "claim" if "claim" in response.lower() else "premise"
            log().debug(f"sentence: {sentence} | predicted as {adu_type}")
            adu : ArgumentUnit = ArgumentUnit(uuid=uuid4(),text=sentence,type=adu_type)
            argument_mining_list.append(adu)

        return argument_mining_list

    
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.
        :param linked_argument_units: The structured links between claims and premises.
        :param originalText: The original input text (for context only).
        :return: LinkedArgumentUnitsWithStance object representing the final stance graph.
        """
        result_linked_arguments: List[StanceRelation] = []
    
        for relation in linked_argument_units.claims_premises_relationships:
            # Find the claim object
            claim = next((c for c in linked_argument_units.claims if c.uuid == relation.claim_id), None)
            if claim is None:
                log().warning(f"No Claim found for this relationship: {relation} --> Continue looping")
                continue
    
            log().debug(f"Claim: {claim.text}")
    
            if not relation.premise_ids:
                log().warning(f"No premises found for this claim ---> continue looping")
                continue
    
            for pid in relation.premise_ids:
                premise = next((p for p in linked_argument_units.premises if p.uuid == pid), None)
                if not premise:
                    log().warning(f"No premise found for the id {pid} ---> continue looping")
                    continue
    
                log().debug(f"  → Premise: {premise.text}")
                prompt = f"""You are given a claim and evidence. Determine whether the evidence supports(pro) or refutes(con) the claim.
    Respond only with one word: "pro" or "con".
    Claim: {claim.text}
    Evidence: {premise.text}
    Stance:"""
    
                result = self.run_prompt(prompt)
    
                # Normalize result to 'pro' or 'con'
                result_lower = result.lower().strip()
                if any(x in result_lower for x in ("refute", "con")):
                    relationship = "con"
                elif any(x in result_lower for x in ("support", "pro")):
                    relationship = "pro"
                else:
                    log().warning(f"Unexpected stance output: {result} — skipping")
                    relationship = "unidentified"
    
                log().debug(f"Claim: {claim.text} | Premise: {premise.text} -> Relationship: {relationship}")
                result_linked_arguments.append(
                    StanceRelation(
                        claim_id=claim.uuid,
                        premise_id=premise.uuid,
                        stance=relationship
                    )
                )
    
        return LinkedArgumentUnitsWithStance(
            original_text=originalText,
            claims=linked_argument_units.claims,
            premises=linked_argument_units.premises,
            stance_relations=result_linked_arguments
        )
    
def test_model(): 
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
    #Test Model
    tinyllamamodel  = TinyLLamaLLMClassifier()
    result = tinyllamamodel.classify_stance(linked_argument_units_test,original_text)
    log().debug(result)