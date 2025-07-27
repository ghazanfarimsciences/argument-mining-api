# app/services/model_client.py

from app.argmining.implementations.encoder_model_loader import MODEL_CONFIGS, NonTrainedEncoderModelLoader, PeftEncoderModelLoader
from app.argmining.implementations.openai_claim_premise_linker import OpenAIClaimPremiseLinker
from app.argmining.implementations.openai_llm_classifier import OpenAILLMClassifier
from app.argmining.implementations.tinyllama_llm_classifier import TinyLLamaLLMClassifier

from app.argmining.interfaces.adu_and_stance_classifier import AduAndStanceClassifier
from app.argmining.models.argument_units import UnlinkedArgumentUnits, LinkedArgumentUnitsWithStance
from ...log import log 

_model_instances: dict[str, AduAndStanceClassifier] = {}

def get_adu_classifier(model_name: str) -> AduAndStanceClassifier:
    if model_name not in _model_instances:
        if model_name == "modernbert":
            model_config = MODEL_CONFIGS.get("modernbert")
            if not model_config:
                raise ValueError("Model configuration for 'modernbert' is not defined.")
            _model_instances[model_name] = PeftEncoderModelLoader(**model_config["params"])
        if model_name == "deberta":
            model_config = MODEL_CONFIGS.get("deberta")
            if not model_config:
                raise ValueError("Model configuration for 'deberta' is not defined.")
            _model_instances[model_name] = NonTrainedEncoderModelLoader(**model_config["params"])
        elif model_name == "openai":
            _model_instances[model_name] = OpenAILLMClassifier()
        elif model_name == "tinyllama":
            _model_instances[model_name] = TinyLLamaLLMClassifier()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    return _model_instances[model_name]

def serialize_linked_argument_units_with_stance(obj: LinkedArgumentUnitsWithStance) -> dict:
    return {
        "original_text": obj.original_text,
        "claims": [
            {
                "id": str(claim.uuid),
                "text": claim.text,
            } for claim in obj.claims
        ],
        "premises": [
            {
                "id": str(premise.uuid),
                "text": premise.text,
            } for premise in obj.premises
        ],
        "stance_relations": [
            {
                "claim_id": str(relation.claim_id),
                "premise_id": str(relation.premise_id),
                "stance": str(relation.stance)
            } for relation in obj.stance_relations
        ]
    }

def run_argument_mining(model_name: str, text: str):
    try:
        log().info("====================== Step1: ADUs classification ======================")
        model = get_adu_classifier(model_name)
        
        unlinked_adus = model.classify_adus(text)
        
        # Link claims and premises using a separate linker

        linked_adus = OpenAIClaimPremiseLinker().link_claims_to_premises(unlinked_adus)
        log().debug(f"Linked ADUs are: {linked_adus}")
        log().info("====================== Step3: Classify Stances ======================")
        # Classify stance
        result = model.classify_stance(linked_adus, text)
        
        result_api = serialize_linked_argument_units_with_stance (result)
        return result_api

    except Exception as e:
        raise RuntimeError(f"Pipeline failed for model '{model_name}': {str(e)}")
