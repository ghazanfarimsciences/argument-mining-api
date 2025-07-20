# app/services/model_client.py

from app.argmining.implementations.encoder_model_loader import EncoderModelLoader
from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier

from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from app.argmining.models.argument_units import UnlinkedArgumentUnits

_model_instances: dict[str, AduAndStanceClassifier] = {}

def get_adu_classifier(model_name: str) -> AduAndStanceClassifier:
    if model_name not in _model_instances:
        if model_name == "modernbert":
            _model_instances[model_name] = EncoderModelLoader()
        elif model_name == "openai":
            _model_instances[model_name] = OpenAILLMClassifier()
        elif model_name == "tinyllama":
            _model_instances[model_name] = TinyLLamaLLMClassifier()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    return _model_instances[model_name]

#TODO: Change it s location from here (Split of concerns)
def convert_to_unlinked_adus(adus):
            return UnlinkedArgumentUnits(
                claims=[adu for adu in adus if adu.type == 'claim'],
                premises=[adu for adu in adus if adu.type == 'premise']
            )

def run_argument_mining(model_name: str, text: str):
    try:
        model = get_adu_classifier(model_name)

        adus = model.classify_adus(text)

        unlinked_adus = convert_to_unlinked_adus(adus)
        # Link claims and premises using a separate linker

        linked_adus = OpenAIClaimPremiseLinker().link_claims_to_premises(unlinked_adus)

        # Classify stance
        result = model.classify_stance(linked_adus, text)

        return result

    except Exception as e:
        raise RuntimeError(f"Pipeline failed for model '{model_name}': {str(e)}")
