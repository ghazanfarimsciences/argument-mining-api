import argparse
import torch
import bitsandbytes as bnb
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from transformers.utils.quantization_config import BitsAndBytesConfig

# Shared 4-bit quantization config (if using 4-bit models)
_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# --- Model loader functions ---
def load_mistral():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_quant_config,
        device_map="auto"
    )
    return tok, model


def load_tinyllama():
    base_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return tok, model


def load_modernbert():
    model_id = "answerdotai/ModernBERT-base"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "premise", 1: "claim"},
        label2id={"premise": 0, "claim": 1}
    )
    return tok, model


def load_roberta():
    model_id = "roberta-base"
    tok = RobertaTokenizer.from_pretrained(model_id)
    model = RobertaForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "premise", 1: "claim"},
        label2id={"premise": 0, "claim": 1}
    )
    return tok, model


def load_deberta_stance():
    model_id = "microsoft/deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        id2label={0: "con", 1: "pro"},
        label2id={"con": 0, "pro": 1}
    )
    return tok, model


# Map names to loader functions\
MODEL_LOADERS = {
    "mistral":       load_mistral,
    "tinyllama":     load_tinyllama,
    "modernbert":    load_modernbert,
    "roberta":       load_roberta,
    "deberta-stance": load_deberta_stance,
}


def main():
    parser = argparse.ArgumentParser(
        description="Choose a model to load and run inference"
    )
    parser.add_argument(
        "--model", choices=list(MODEL_LOADERS.keys()), required=True,
        help="Model to use"
    )
    parser.add_argument(
        "--prompt", type=str, help="Prompt (for generative models)"
    )
    parser.add_argument(
        "--text", type=str, help="Text to classify (for encoder models)"
    )
    args = parser.parse_args()

    tok, model = MODEL_LOADERS[args.model]()

    if args.model in ["mistral", "tinyllama"]:
        # Generative decoding
        prompt = args.prompt or input("Enter prompt: ")
        inputs = tok(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        print(tok.decode(outputs[0], skip_special_tokens=True))
    else:
        # Sequence classification
        text = args.text or input("Enter text to classify: ")
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        logits = model(**inputs).logits
        pred_id = int(torch.argmax(logits, dim=-1))
        label = model.config.id2label[pred_id]
        print(f"Predicted label: {label}")

# Has to be adjusted, currently as CLI for testing purposes
if __name__ == "__main__":
    main()
