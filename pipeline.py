from langgraph.graph import StateGraph, END
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
import logging
import os
import matplotlib.pyplot as plt

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filemode="a"
)

# Main fine-tuned model
MODEL_PATH = "my_distilbert_imdb_model"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Backup zero-shot classifier for fallback
backup_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def backup_model_predict(text):
    candidate_labels = ["positive", "negative"]
    result = backup_classifier(text, candidate_labels)
    label = result["labels"][0]
    confidence = result["scores"][0]
    return (1 if label == "positive" else 0, confidence)

State = dict

# Inference node
def inference_node(state):
    text = state["input"]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    confidence, pred = probs.max(dim=1)
    state["predicted_label"] = int(pred.item())
    state["confidence"] = float(confidence.item())
    logging.info(f"[Inference] Input='{text}' | Pred={state['predicted_label']} | Conf={state['confidence']:.2f}")
    return state

# Confidence check node
def confidence_check_node(state, threshold=0.7):
    state["fallback"] = state["confidence"] < threshold
    logging.info(f"[ConfidenceCheck] Score={state['confidence']:.2f} | Fallback={state['fallback']}")
    return state

# Fallback node with backup model and user clarification
def fallback_node(state):
    print(f"[FallbackNode] Low confidence ({state.get('confidence', 0)*100:.1f}%). Trying backup model...")
    backup_label, backup_conf = backup_model_predict(state["input"])
    print(f"[BackupModel] Prediction: {'Positive' if backup_label else 'Negative'} ({backup_conf*100:.1f}% confidence)")
    user_input = input("Keep backup label or clarify manually? (keep/clarify): ").strip().lower()
    if "clarify" in user_input:
        clarification = input("Was this a positive or negative review? ").strip().lower()
        state["final_label"] = 1 if "positive" in clarification else 0
        logging.info(f"[Fallback] User clarified to {state['final_label']}")
    else:
        state["final_label"] = backup_label
        logging.info(f"[Fallback] Kept backup model label {state['final_label']} with conf {backup_conf}")
    return state

# Building LangGraph
graph = StateGraph(State)
graph.add_node("inference", inference_node)
graph.add_node("confidence_check", confidence_check_node)
graph.add_node("fallback", fallback_node)
graph.set_entry_point("inference")
graph.add_edge("inference", "confidence_check")
graph.add_conditional_edges(
    "confidence_check",
    lambda s: not s["fallback"],
    path_map={True: END, False: "fallback"}
)
graph.add_edge("fallback", END)
app = graph.compile()

# Track confidence and fallback count
confidence_history = []
fallback_count = 0

def print_confidence_curve(history):
    print("\nConfidence Scores per Input:")
    for i, c in enumerate(history, 1):
        print(f"Input {i}: {c:.2f}")

def print_fallback_count(count):
    print("\nFallback Trigger Count:")
    print(count)

def save_metrics_plots(history, fallback_count):
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 4))
    # Confidence curve 
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(history) + 1), history, marker='o', color='b')
    plt.title("Confidence Curve")
    plt.xlabel("Input #")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    # Fallback frequency 
    plt.subplot(1, 2, 2)
    plt.bar(['Fallbacks'], [fallback_count], color='orange')
    plt.title("Fallback Frequency")
    plt.ylabel("Count")
    plt.ylim(0, max(1, fallback_count + 1))
    plt.tight_layout()
    plt.savefig("results/session_metrics.png")
    plt.close()

# CLI loop with tracking
if __name__ == "__main__":
    while True:
        user_text = input("\nEnter review (or 'exit' to quit): ").strip()
        if user_text.lower() == "exit":
            print("\nExiting CLI.")
            print_confidence_curve(confidence_history)
            print_fallback_count(fallback_count)
            save_metrics_plots(confidence_history, fallback_count)
            print("Session metrics plot saved as results/session_metrics.png")
            break

        state = {"input": user_text}
        result = app.invoke(state)

        confidence_history.append(result.get("confidence", 0))
        if result.get("fallback", False):
            fallback_count += 1

        label = "Positive" if result["predicted_label"] == 1 else "Negative"
        print(f"[InferenceNode] Predicted label: {label} | Confidence: {result['confidence']*100:.0f}%")

        if "final_label" in result:
            final_label = "Positive" if result["final_label"] == 1 else "Negative"
            print(f"[Final Label] {final_label} (via fallback)")
        else:
            print(f"[Final Label] {label}")

        logging.info(f"[RESULT] Input: {user_text} | Final label: {result.get('final_label', result['predicted_label'])}")




