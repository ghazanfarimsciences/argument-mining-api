from dataclasses import asdict
import json
import uuid
import openai
import torch
import re 
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
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
        self.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.adapter_path = os.path.join(os.path.dirname(__file__), "TinyLlama-1.1B-Chat-v1.0_finetuned")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_auth_token=HF_TOKEN)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id, 
            use_auth_token=HF_TOKEN,
            torch_dtype=torch.float16,  # Use fp16 for efficiency
            device_map="auto"
        )
        
        # Load PEFT adapter
        try:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            log().info(f"Successfully loaded PEFT adapter from {self.adapter_path}")
        except Exception as e:
            log().warning(f"Failed to load PEFT adapter from {self.adapter_path}: {e}")
            log().warning("Falling back to base model")
            self.model = base_model
    
    def run_prompt(self, prompt, max_new_tokens=150, max_retries=3, retry_delay=1):
        """
        Runs the prompt on the model with a retry mechanism.
        Args:
            prompt (str): The prompt to send to the model.
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
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.3
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
    
    def _parse_adu_output(self, output_text: str) -> List[dict]:
        """
        Parses the model output to extract ADUs with their types.
        Expected format: CLAIM: text... PREMISE: text... etc.
        """
        adus = []
        lines = output_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for CLAIM: or PREMISE: patterns
            if line.upper().startswith('CLAIM:'):
                text = line[6:].strip()
                if text:
                    adus.append({'type': 'claim', 'text': text})
            elif line.upper().startswith('PREMISE:'):
                text = line[8:].strip()
                if text:
                    adus.append({'type': 'premise', 'text': text})
            # Alternative patterns
            elif '- CLAIM:' in line.upper():
                text = line.split('- CLAIM:', 1)[1].strip()
                if text:
                    adus.append({'type': 'claim', 'text': text})
            elif '- PREMISE:' in line.upper():
                text = line.split('- PREMISE:', 1)[1].strip()
                if text:
                    adus.append({'type': 'premise', 'text': text})
        
        return adus
        
    def classify_adus(self, text: str) -> UnlinkedArgumentUnits:
        """
        Extracts and labels argumentative units from text using full context.
        Args:
            text (str): The raw input document.
        Returns:
            UnlinkedArgumentUnits: List of claims and premises extracted from the text.
        """
        log().info(f"Analyzing text with context-aware approach")
        
        # Context-aware prompt that considers the entire text
        prompt = f"""
You are an expert argument-mining classifier. Analyze the following text and identify all argumentative discourse units (ADUs) within their full context.

Task: 
1. Read the ENTIRE text to understand the context and argument structure
2. Identify argumentative units that function as either CLAIMS or PREMISES
3. Consider how each unit relates to the overall argument structure
4. Extract meaningful argumentative segments (can be phrases, sentences, or sentence groups)

Definitions:
- CLAIM: A statement that takes a stance, makes an assertion, or expresses what should/shouldn't happen
- PREMISE: A statement that provides evidence, reasons, data, examples, or explanations to support or refute claims

Context Guidelines:
- Consider the rhetorical function of each unit within the broader argument
- Look for implicit argumentative relationships between statements
- Identify units that work together to form coherent argumentative moves
- Pay attention to discourse markers and connectives that signal argumentative relationships

Output Format:
For each identified ADU, write on a new line:
CLAIM: [the exact text of the claim]
or
PREMISE: [the exact text of the premise]

Text to analyze:
"{text}"

Identified ADUs:
"""

        response = self.run_prompt(prompt, max_new_tokens=300)
        log().debug(f"Raw model response: {response}")
        
        # Parse the response to extract ADUs
        parsed_adus = self._parse_adu_output(response)
        log().info(f"Parsed {len(parsed_adus)} ADUs from model response")
        
        # Initialize separate lists for claims and premises
        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []
        
        # Convert parsed ADUs to ArgumentUnit objects
        for adu_data in parsed_adus:
            adu = ArgumentUnit(
                uuid=uuid4(), 
                text=adu_data['text'], 
                type=adu_data['type']
            )
            
            if adu_data['type'] == 'claim':
                claims.append(adu)
                log().debug(f"Identified CLAIM: {adu_data['text']}")
            else:
                premises.append(adu)
                log().debug(f"Identified PREMISE: {adu_data['text']}")
        
        # Fallback: If no ADUs were extracted, try sentence-by-sentence with context
        if not claims and not premises:
            log().warning("No ADUs extracted with context approach, falling back to enhanced sentence analysis")
            return self._fallback_sentence_analysis(text)
            
        return UnlinkedArgumentUnits(claims=claims, premises=premises)
    
    def _fallback_sentence_analysis(self, text: str) -> UnlinkedArgumentUnits:
        """
        Fallback method that analyzes sentences but with full text context.
        """
        sentences = split_into_sentences(text)
        log().info(f"Fallback: Found {len(sentences)} sentences in the input text")

        claims: List[ArgumentUnit] = []
        premises: List[ArgumentUnit] = []

        for i, sentence in enumerate(sentences):
            # Create context window (previous and next sentences)
            context_before = ' '.join(sentences[max(0, i-2):i]) if i > 0 else ""
            context_after = ' '.join(sentences[i+1:min(len(sentences), i+3)]) if i < len(sentences)-1 else ""
            
            prompt = f"""
You are an argument-mining classifier analyzing a sentence within its context.

Full Text Context: "{text}"

Previous Context: "{context_before}"
Current Sentence: "{sentence}"
Following Context: "{context_after}"

Task: Classify the current sentence as either "claim" or "premise" considering its role in the broader argument.

Definitions:
- claim: a statement that takes a stance or asserts something to be true/false or should/shouldn't happen
- premise: a statement that gives evidence, reasons, data, or explanation intended to support/refute a claim

Rules:
- Consider the sentence's function within the overall argumentative structure
- Output EXACTLY ONE lowercase word: "claim" or "premise"
- Do NOT add punctuation or extra text

Answer:
"""

            response = self.run_prompt(prompt, max_new_tokens=10)
            adu_type = "claim" if "claim" in response.lower() else "premise"
            
            log().debug(f"Sentence: {sentence} | Predicted as: {adu_type}")
            
            adu = ArgumentUnit(uuid=uuid4(), text=sentence, type=adu_type)
            if adu_type == "claim":
                claims.append(adu)
            else:
                premises.append(adu)
                
        return UnlinkedArgumentUnits(claims=claims, premises=premises)

    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.
        Now uses the original text context for better stance classification.
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
                
                # Enhanced prompt with context
                prompt = f"""You are analyzing the relationship between a claim and evidence within a broader argumentative context.

Original Text Context: "{originalText}"

Claim: "{claim.text}"
Evidence: "{premise.text}"

Task: Determine whether the evidence supports (pro) or refutes (con) the claim, considering the broader context.

Rules:
- "pro" means the evidence supports, strengthens, or provides reasons for the claim
- "con" means the evidence refutes, weakens, or provides reasons against the claim
- Consider the argumentative context and how these elements function together
- Respond with exactly one word: "pro" or "con"

Stance:"""
    
                result = self.run_prompt(prompt, max_new_tokens=10)
    
                # Normalize result to 'pro' or 'con'
                result_lower = result.lower().strip()
                if any(x in result_lower for x in ("refute", "con", "against", "oppose")):
                    relationship = "con"
                elif any(x in result_lower for x in ("support", "pro", "for", "favor")):
                    relationship = "pro"
                else:
                    log().warning(f"Unexpected stance output: {result} — defaulting to 'pro'")
                    relationship = "pro"  # Default assumption
    
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
    log().info(f"--- Running Step 1: Context-Aware ADU Classification using TinyLLama ---")
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
    log().info(f"--- Running Step 3: Context-Aware Stance Classification using TinyLLama ---")
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