import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "my_distilbert_imdb_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def inference_node(state):
    if "input" not in state:
        raise KeyError("Missing key 'input' in state. Available keys: " + ", ".join(state.keys()))
    text = state["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    confidence, pred = probs.max(dim=1)
    state["predicted_label"] = int(pred.item())
    state["confidence"] = float(confidence.item())
    return state
