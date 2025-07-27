from dataclasses import asdict
import json
import openai
import re 

from uuid import UUID, uuid4

from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance, StanceRelation, ClaimPremiseRelationship, UnlinkedArgumentUnits
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
        self.system_prompt_adu_classification = """            You are an argument-mining classifier.

Task: Decide whether the SINGLE input sentence is a **claim** or a **premise**.

Definitions (use these only):
- claim: a statement that takes a stance or asserts something to be true/false or should/shouldn't happen.
- premise: a statement that gives evidence, reasons, data, or explanation intended to support/refute a claim.

Rules:
- Output EXACTLY ONE lowercase word: "claim" or "premise".
- If the sentence mixes both, pick the main function (assertion → claim; support/explanation → premise).
- Do NOT add punctuation or extra text.

Answer:
                        """
        self.system_prompt_stance_classification = """You are an assistant for argument mining.
                        You are given a claim and evidence (premise).
                        Determine whether the evidence supports ('pro') or refutes ('con') the claim.
                        Respond only with one word: "pro" or "con".
                        """
        
    def classify_sentence(self, sentence: str, model: str = "gpt-4.1") -> str:
        """
        Executes the prompt to classify a single sentence as a 'claim' or 'premise'.
        Includes a retry mechanism with fallback to gpt-3.5-turbo.
        """
        user_prompt = f'Sentence: "{sentence}"\nWhat is it?'
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt_adu_classification.strip(),
                    },
                    {"role": "user", "content": user_prompt.strip()},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip().lower() # type: ignore
        except Exception as e:
            log().warning(f"❌ Model {model} failed for ADU classification: {e}")
            if model != "gpt-3.5-turbo":
                log().info(
                    "Attempting with gpt-3.5-turbo for ADU classification."
                )
                return self.classify_sentence(sentence, model="gpt-3.5-turbo")
            log().error(
                "Failed ADU classification after retries. Defaulting to 'premise'."
            )
            return "premise"  # Fallback if all models fail
        
    def classify_stance_single(self, claim_text: str, premise_text: str, model: str = "gpt-4.1") -> str:
        """
        Classifies the stance between a single claim and premise.
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
            result = response.choices[0].message.content.strip().lower() # type: ignore
            if any(x in result for x in ("refute", "con")):
                return "con"
            elif any(x in result for x in ("support", "pro")):
                return "pro"
            else:
                log().warning(f"Unexpected stance output: {result} for Claim: '{claim_text}' | Premise: '{premise_text}'")
                return "unidentified"
        except Exception as e:
            log().warning(f"❌ Model {model} failed for stance classification: {e}")
            return "unidentified" # Fallback if all models fail

            
    def classify_adus(self, text: str) -> UnlinkedArgumentUnits:
        """
        Extracts and labels argumentative units from text.
        Args:
            text (str): The raw input document.
        Returns:
            UnlinkedArgumentUnits: An object containing lists of claims and premises.
        """
        # Step 1: Split paragraph into sentences
        sentences = split_into_sentences(text)
        log().info(f"Found {len(sentences)} sentences in the input text")

        # Step 2: Classify each sentence as a claim or premise
        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []

        model_to_use = "gpt-4.1"  # Can be configured
        current_pos = 0
        for sentence in sentences:
            if not sentence:  # Skip empty strings that might result from splitting
                continue

            adu_type = self.classify_sentence(sentence, model_to_use)

            start_pos = text.find(sentence, current_pos)
            end_pos = start_pos + len(sentence) if start_pos != -1 else -1

            log().debug(
                f"Sentence: '{sentence}' | Predicted as: {adu_type} | Start: {start_pos} | End: {end_pos}"
            )

            adu = ArgumentUnit(
                uuid=uuid4(),
                text=sentence,
                type=adu_type,
                start_pos=start_pos,
                end_pos=end_pos,
                confidence=None  # Confidence is not used in this implementation
            )

            # Append the new ADU to the correct list
            if adu.type == "claim":
                claims.append(adu)
            else:
                premises.append(adu)

            if start_pos != -1:
                current_pos = end_pos

        # Construct and return the UnlinkedArgumentUnits object
        return UnlinkedArgumentUnits(claims=claims, premises=premises)
    
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.

        :param linked_argument_units: The structured links between claims and premises.
        :param originalText: The original text from which the claims and premises were extracted (for context only).
        :return: LinkedArgumentUnitsWithStance object representing the final stance graph.
        """
        result_linked_arguments: List[StanceRelation] = []
        model_to_use = "gpt-4.1" # Can be configured

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
                        confidence=None # OpenAI doesn't directly provide confidence score in this API call
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
    miner: AduAndStanceClassifier = OpenAILLMClassifier()
    
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