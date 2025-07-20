from pathlib import Path
import torch
import json
import re
from typing import List, Dict
from uuid import UUID as uuid, uuid4
from dataclasses import asdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from peft import PeftModel
import logging

from ..models.argument_units import ArgumentUnit, UnlinkedArgumentUnits

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EncoderModelLoader:
    """
    This class lets you load and use a model like ModernBERT, or any other model, for argument mining tasks.
    It supports ADU identification, ADU classification, and stance classification.
    """
    
    def __init__(self, base_model_path: str, adapter_paths: Dict[str, str], device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.base_model_path = base_model_path
        
        root = Path(__file__).parent
        print(f"Root path: {root}")
        self.adapter_paths = {
            task: str((root / rel_path))
            for task, rel_path in adapter_paths.items()
        }
        print(f"Adapter paths: {self.adapter_paths}")
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # Models loaded on demand
        self.models = {}
        
        # Task configurations
        self.task_configs = {
            'adu_identification': {
                'labels': ['No', 'Yes'],
                'task_type': 'token_classification'
            },
            'adu_classification': {
                'labels': ['claim', 'premise'],
                'task_type': 'sequence_classification'
            },
            'stance_classification': {
                'labels': ['con', 'pro'],
                'task_type': 'sequence_classification'
            }
        }
        
        logger.info("ModernBERT Argument Miner initialized")
    
    def load_model_for_task(self, task_name: str):
        if task_name in self.models:
            return self.models[task_name]
            
        logger.info(f"Loading model for task: {task_name}")
        
        if self.task_configs[task_name]['task_type'] == 'token_classification':
            base_model = AutoModelForTokenClassification.from_pretrained(
                self.base_model_path,
                num_labels=len(self.task_configs[task_name]['labels'])
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.base_model_path,
                num_labels=len(self.task_configs[task_name]['labels'])
            )
        
        model = PeftModel.from_pretrained(base_model, self.adapter_paths[task_name])
        model = model.to(self.device)
        model.eval()
        
        self.models[task_name] = model
        return model
    
    def identify_adus(self, text: str, confidence_threshold: float = 0.5) -> List[ArgumentUnit]:
        model = self.load_model_for_task('adu_identification')
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
            return_offsets_mapping=True
        ).to(self.device)
        
        offset_mapping = inputs.pop('offset_mapping')[0]
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        labels = self.task_configs['adu_identification']['labels']
        pred_labels = [labels[p] for p in predictions[0].cpu().numpy()]
        confidences = probabilities[0, :, 1].cpu().numpy()
        
        adus = []
        current_span = None
        
        for i, (pred_label, confidence, offset) in enumerate(zip(pred_labels, confidences, offset_mapping)):
            if pred_label == 'Yes' and confidence > confidence_threshold:
                if current_span is None:
                    current_span = {
                        'start': offset[0].item(),
                        'end': offset[1].item(),
                        'confidence': confidence
                    }
                else:
                    current_span['end'] = offset[1].item()
                    current_span['confidence'] = max(current_span['confidence'], confidence)
            else:
                if current_span is not None:
                    span_text = text[current_span['start']:current_span['end']].strip()
                    if span_text:
                        adus.append(ArgumentUnit(
                            uuid=uuid4(),
                            text=span_text,
                            start_pos=current_span['start'],
                            end_pos=current_span['end'],
                            type='unknown',
                            confidence=float(current_span['confidence'])
                        ))
                    current_span = None
        
        # handle final span
        if current_span is not None:
            span_text = text[current_span['start']:current_span['end']].strip()
            if span_text:
                adus.append(ArgumentUnit(
                    uuid=uuid4(),
                    text=span_text,
                    start_pos=current_span['start'],
                    end_pos=current_span['end'],
                    type='unknown',
                    confidence=float(current_span['confidence'])
                ))
        
        return adus
    
    def classify_adu_type(self, adu_text: str) -> tuple[str, float]:
        model = self.load_model_for_task('adu_classification')
        
        inputs = self.tokenizer(
            adu_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities).item()
        
        labels = self.task_configs['adu_classification']['labels']
        return labels[prediction.item()], confidence
    
    def classify_stance(self, claim_text: str, premise_text: str) -> tuple[str, float]:
        model = self.load_model_for_task('stance_classification')
        
        combined_text = f"[CLS] {claim_text} [SEP] {premise_text} [SEP]"
        inputs = self.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities).item()
        
        labels = self.task_configs['stance_classification']['labels']
        return labels[prediction.item()], confidence
    
    def extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_text_and_identify_adus(self, text: str, use_sentence_fallback: bool = True) -> UnlinkedArgumentUnits:
        logger.info(f"Processing text of length {len(text)}")
        adus = self.identify_adus(text)
        
        # fallback to sentences
        if not adus and use_sentence_fallback:
            logger.info("No ADUs identified, using sentence-level fallback")
            sentences = self.extract_sentences(text)
            adus = []
            current_pos = 0
            for sentence in sentences:
                start_pos = text.find(sentence, current_pos)
                if start_pos != -1:
                    adus.append(ArgumentUnit(
                        uuid=uuid4(),
                        text=sentence,
                        start_pos=start_pos,
                        end_pos=start_pos + len(sentence),
                        type='unknown',
                        confidence=0.5  # Default confidence for fallback sentences
                    ))
                    current_pos = start_pos + len(sentence)
        
        # classify ADU types
        claims : List[ArgumentUnit] = []
        premises : List[ArgumentUnit] = []
        for adu in adus:
            adu_type, conf = self.classify_adu_type(adu.text)
            adu.type = adu_type
            adu.confidence = min(adu.confidence, conf)
            (claims if adu_type == 'claim' else premises).append(adu)
        
        return UnlinkedArgumentUnits(premises=premises, claims=claims)

        

        #### Note : Andre This can not be done before the relationship between claims and premise is known. ####
        # # build stance relations
        # stance_relations: List[StanceRelation] = []
        # for claim in claims:
        #     for premise in premises:
        #         stance, conf = self.classify_stance(claim.text, premise.text)
        #         stance_relations.append(StanceRelation(
        #             claim_id=claim.uuid,
        #             premise_id=premise.uuid,
        #             stance=stance,
        #             confidence=conf
        #         ))
        
        # return ArgumentStructure(
        #     original_text=text,
        #     claims=claims,
        #     premises=premises,
        #     stance_relations=stance_relations
        # )
    
    def process_json_input(self, json_input: str) -> str:
        try:
            input_data = json.loads(json_input)
            text = input_data.get('text', '')
            if not text:
                raise ValueError("No 'text' field found in input JSON")
            
            result = self.process_text_and_identify_adus(text)
            
            # serialize dataclass → dict → JSON, converting UUIDs to strings
            def _convert(o):
                if isinstance(o, uuid): return str(o)
                if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
                if isinstance(o, list): return [_convert(v) for v in o]
                return o
            
            raw = asdict(result)
            clean = _convert(raw)
            return json.dumps(clean, indent=2)
            
        except Exception as e:
            logger.error(f"Error processing JSON input: {str(e)}")
            return json.dumps({'error': str(e), 'status': 'failed'}, indent=2)
            
class DebertaModelLoader:
    def __init__(self, type_model_path: str, stance_model_path: str):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(type_model_path)
        self.type_model = AutoModelForSequenceClassification.from_pretrained(type_model_path).to(self.device)
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(stance_model_path).to(self.device)
        self.type_model.eval()
        self.stance_model.eval()

    def extract_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def classify_adu_type(self, sentence: str) -> tuple[str, float]:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.type_model(**inputs)
            probs = torch.softmax(output.logits, dim=-1)
            pred = torch.argmax(probs).item()
            conf = torch.max(probs).item()
        label = 'claim' if pred == 1 else 'premise'
        return label, conf

    def classify_stance(self, claim_text: str, premise_text: str) -> tuple[str, float]:
        combined = f"{claim_text} [SEP] {premise_text}"
        inputs = self.tokenizer(combined, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            output = self.stance_model(**inputs)
            probs = torch.softmax(output.logits, dim=-1)
            pred = torch.argmax(probs).item()
            conf = torch.max(probs).item()
        label = 'pro' if pred == 1 else 'con'
        return label, conf

    def process_text(self, text: str) -> UnlinkedArgumentUnits:
        sentences = self.extract_sentences(text)
        claims, premises = [], []
        current_pos = 0

        for sentence in sentences:
            start = text.find(sentence, current_pos)
            end = start + len(sentence)
            current_pos = end
            adu_type, conf = self.classify_adu_type(sentence)
            adu = ArgumentUnit(uuid=uuid4(), text=sentence, start_pos=start, end_pos=end, type=adu_type, confidence=conf)
            (claims if adu_type == 'claim' else premises).append(adu)

        return UnlinkedArgumentUnits(claims=claims, premises=premises)

MODEL_AND_ADAPTERS = {
    "ModernBERT": {
        "base_model_path": "answerdotai/ModernBERT-base",
        "adapter_paths": {
            'adu_identification':   "argument-mining-modernBert/argument-mining-modernbert-adu_identification/checkpoint-9822",
            'adu_classification':   "argument-mining-modernBert/argument-mining-modernbert-adu_classification/checkpoint-9822",
            'stance_classification':"argument-mining-modernBert/argument-mining-modernbert-stance_classification/checkpoint-4911",
        }
    },
{
"DeBERTa": {
    "base_model_path": "microsoft/deberta-v3-base",
    "type_model_path": "models/deberta-adu-type/checkpoint-3000",
    "stance_model_path": "models/deberta-stance/checkpoint-2800"
}

}
}

def test_modern_bert():
    miner = EncoderModelLoader(
        base_model_path=MODEL_AND_ADAPTERS["ModernBERT"]["base_model_path"],
        adapter_paths=MODEL_AND_ADAPTERS["ModernBERT"]["adapter_paths"],
    )
    
    example_input = {
        "text": "Climate Change is made up. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."
    }
    
    json_input = json.dumps(example_input)
    result = miner.process_json_input(json_input)
    
    print("Input:")
    print(json.dumps(example_input, indent=2))
    print("\nOutput:")
    print(result)

def test_deberta():
    model_config = MODEL_AND_ADAPTERS["DeBERTa"]

    miner = DebertaModelLoader(
        type_model_path=model_config["type_model_path"],
        stance_model_path=model_config["stance_model_path"]
    )

    example_input = {
        "text": "Climate Change is made up. Urban gardening is not just a trend; it is a necessary adaptation to modern urban life. Cities are increasingly crowded, and access to fresh produce is often limited in low-income neighborhoods. By turning rooftops, balconies, and vacant lots into green spaces, residents can take control of their food sources. This not only improves nutrition but also promotes community building and environmental awareness. Moreover, urban gardens help reduce the urban heat island effect, making cities more livable during extreme weather. While some argue that the scale of urban gardening is too small to make a real impact, its cumulative effects—both social and ecological—can be profound."
    }

    json_input = json.dumps(example_input)
    input_text = json.loads(json_input)["text"]
    result = miner.process_text(input_text)

    print("Input:")
    print(json.dumps(example_input, indent=2))

    output_dict = {
        "claims": [claim.text for claim in result.claims],
        "premises": [premise.text for premise in result.premises]
    }

    print("\nOutput:")
    print(json.dumps(output_dict, indent=2))

if __name__ == "__main__":
    test_modern_bert()
