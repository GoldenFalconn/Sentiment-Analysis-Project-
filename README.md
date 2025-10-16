# Self-Healing Classification DAG (LangGraph + Transformers)

A sentiment classification pipeline with a fine-tuned transformer (DistilBERT) and confidence-based fallback using LangGraph. Supports user input fallback and backup zero-shot classifier with structured logging and visualization.

---

## Features

- Fine-tuned DistilBERT model on IMDB sentiment dataset
- LangGraph pipeline with inference, confidence check, and fallback nodes
- Backup zero-shot fallback model (facebook/bart-large-mnli)
- CLI interface with interactive clarification on low-confidence predictions
- Structured logging under `logs/pipeline.log`
- Confidence curve and fallback frequency tracking with CLI and matplotlib visualization

---

## Folder Structure

.
├── logs/
│ └── pipeline.log
├── my_distilbert_imdb_model/
├── nodes/
│ ├── inference_node.py
│ ├── confidence_check_node.py
│ └── fallback_node.py
├── pipeline.py
├── requirements.txt
├── README.md
├── training.ipynb
├── results/


---

## Setup

1. Clone repo and move inside folder  
2. Install dependencies:  
pip install -r requirements.txt

3. Ensure model files are inside `my_distilbert_imdb_model/` or download using instructions.

---
## Dataset Used
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Usage

Run the pipeline:

python pipeline.py


Follow prompts to enter reviews or type `exit` to quit and view confidence and fallback stats.

---

## Fallback Details

- Confidence below 0.7 triggers fallback
- Backup zero-shot classifier suggests a label
- User can confirm or clarify to provide final label

---

## Logging and Visualization

- Logs saved in `logs/pipeline.log`
- On exit, textual confidence scores and fallback counts shown
- Optional matplotlib charts display confidence curve and fallback frequency if matplotlib is installed

---

## Model Fine-Tuning

See `training.ipynb` for full training and export process on IMDB dataset.

---

## Demo Video

Included as `demo_video.mp4` or linked in `demo_link.txt`, demonstrating CLI workflow, fallback logic, and visualizations.

---

## Contact

For questions or issues, please open an issue or email your.email@example.com

Follow prompts to enter reviews or type `exit` to quit and view confidence and fallback stats.

---

## Fallback Details

- Confidence below 0.7 triggers fallback
- Backup zero-shot classifier suggests a label
- User can confirm or clarify to provide final label

---

## Logging and Visualization

- Logs saved in `logs/pipeline.log`
- On exit, textual confidence scores and fallback counts shown
- Optional matplotlib charts display confidence curve and fallback frequency if matplotlib is installed

---

## Model Fine-Tuning

See `training.ipynb` for full training and export process on IMDB dataset.

---

## Demo Video

Included as `demo_video.mp4` or linked in `demo_link.txt`, demonstrating CLI workflow, fallback logic, and visualizations.

---

## Contact

Feel free to reach out to me in case of any queries or feedback at: 
work.swatisingh12@gmail.com
