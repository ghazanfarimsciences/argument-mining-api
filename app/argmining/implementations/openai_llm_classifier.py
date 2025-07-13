import openai
import re 

from uuid import UUID, uuid4
from ..interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from ..models.argument_units import ArgumentUnit, LinkedArgumentUnits, LinkedArgumentUnitsWithStance
from typing import List
from ..config import OPENAI_KEY
from ..log import log


#TODO: Implement Pydantic
#NOTE: classify_stance is not implement yet 


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())
      
class OpenAILLMClassifier (AduAndStanceClassifier): 
    def __init__ (self): 
        self.client = openai.OpenAI(api_key=OPENAI_KEY)
        self.system_prompt = """You are an assistant for argument mining.
                        Classify the following sentence as either a 'claim' or a 'premise'.
                        Only return one word: claim or premise.
                        """
        
    def classify_sentence (self,sentence,model): 
        """executing the prompt to classify a single sentence """
        user_prompt = f'Sentence: "{sentence}"\nWhat is it?'
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.system_prompt.strip()},
                    {"role": "user", "content": user_prompt.strip()}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            log().warning(f"❌ Model {model} failed: {e}")
            if model != "gpt-3.5-turbo":
                return self.classify_sentence(sentence, model="gpt-3.5-turbo")
            return "premise"  # NOTE: Check if this one makes sence     
        
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

        #Step2: predict ADU Type for every sentence 
        argument_mining_list = []
        model = "gpt-4-turbo"
        for sentence in sentences : 
            adu_type = self.classify_sentence(sentence,model)
            log().debug(f"sentence: {sentence} | predicted as {adu_type}")
            adu : ArgumentUnit = ArgumentUnit(uuid=uuid4(),text=sentence,type=adu_type)
            argument_mining_list.append(adu)

        return argument_mining_list
    
    def classify_stance(self, linked_argument_units: LinkedArgumentUnits, originalText: str) -> LinkedArgumentUnitsWithStance:
        """
        Classifies the stance of argument units (ADUs) and links claims to premises.

        :param claims: List of claims to be linked.
        :param premises: List of premises to be linked.
        :param originalText: The original text from which the claims and premises were extracted (Nothing is done with it, just passed for context).
        :return: A list of relationships between claims and premises with their stance.
        """   
