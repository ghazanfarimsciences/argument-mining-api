from dataclasses import asdict
import json
import uuid
import openai
import torch
import re 

from transformers import AutoTokenizer, AutoModelForCausalLM
from uuid import UUID, uuid4

from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship, UnlinkedArgumentUnits
from typing import List
from ...log import log
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
        
        
    def classify_adus(self, text: str) -> UnlinkedArgumentUnits:
        """Extracts and labels argumentative units from text.
        Args:
            text (str):  The raw input document.
        Returns:
            UnlinkedArgumentUnits: List of claims and premises extracted from the text.
        """
        # Step 1: Split paragraph into sentences
        sentences = split_into_sentences(text)
        log().info(f"Found {len(sentences)} in the input text")

        # Initialize separate lists for claims and premises
        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []

        # Step 2: Iterate through sentences and classify them
        for sentence in sentences:
            prompt = f"""
            You are an argument-mining classifier.

Task: Decide whether the SINGLE input sentence is a **claim** or a **premise**.

Definitions (use these only):
- claim: a statement that takes a stance or asserts something to be true/false or should/shouldn't happen.
- premise: a statement that gives evidence, reasons, data, or explanation intended to support/refute a claim.

Rules:
- Output EXACTLY ONE lowercase word: "claim" or "premise".
- If the sentence mixes both, pick the main function (assertion → claim; support/explanation → premise).
- Do NOT add punctuation or extra text.

Sentence: "{sentence}"

Answer:
            """

            response = self.run_prompt(prompt)
            adu_type = (
                "claim" if "claim" in response.lower() else "premise"
            )
            log().debug(f"sentence: {sentence} | predicted as {adu_type}")
            adu: ArgumentUnit = ArgumentUnit(
                uuid=uuid4(), text=sentence, type=adu_type
            )
            if adu_type == "claim":
                claims.append(adu)
            else:
                premises.append(adu)
        return UnlinkedArgumentUnits(claims=claims, premises=premises)

    
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
    """
    Tests the full pipeline for the TinyLlama LLM Classifier.
    This includes classifying ADUs, linking claims to premises, and classifying stance.
    """
    
    # Factory: Instantiate the correct class with its specific parameters
    miner: AduAndStanceClassifier = TinyLLamaLLMClassifier()
    
    claim_linker = OpenAIClaimPremiseLinker()
    example_text = "Climate Change is made up. The measurements of temperature were only recorded the last 100 years, before that there could've been even hotter times. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."

    # The rest of the pipeline is IDENTICAL for both models because they share the same interface.
    
    # --- Step 1: Classify ADUs to get unlinked claims and premises ---
    (f"--- Running Step 1: Classify ADUs using TinyLLama ---")
    unlinked_adus = miner.classify_adus(example_text)
    log().info(f"Found Claims: {len(unlinked_adus.claims)}")
    log().info(f"Found Premises: {len(unlinked_adus.premises)}")
    log().info("--------------------")

    # --- Step 2: Link claims to premises using the OpenAI linker ---
    log().info("--- Running Step 2: Linking Claims to Premises (OpenAI) ---")
    try:
        linked_adus = claim_linker.link_claims_to_premises(unlinked_adus)
        log().info(f"Successfully linked ADUs.")
        log().info("--------------------")
    except (openai.AuthenticationError, ValueError) as e:
        log().error(f"ERROR: Could not run linking step. Please check your OpenAI API key. Details: {e}")
        return
    except Exception as e:
        log().error(f"An unexpected error occurred during linking: {e}")
        return

    # --- Step 3: Classify the stance for the linked units ---
    log().info(f"--- Running Step 3: Classify Stance using TinyLLama ---")
    final_structure = miner.classify_stance(
        linked_argument_units=linked_adus, originalText=example_text
    )
    log().info(f"Generated {len(final_structure.stance_relations)} stance relations.")
    log().info("--------------------")

    # --- Final Output ---
    def _convert_to_json(o):
        if isinstance(o, UUID): return str(o)
        if isinstance(o, dict): return {k: _convert_to_json(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert_to_json(v) for v in o]
        return o

    raw_dict = asdict(final_structure)
    final_json = json.dumps(_convert_to_json(raw_dict), indent=2)

    log().info("\n--- Final Argument Structure Output ---")
    log().info(final_json)