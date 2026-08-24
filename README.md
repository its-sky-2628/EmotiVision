# Sentiment Analysis — End-to-End NLP Project

A complete, runnable NLP project: synthetic review dataset generation,
text preprocessing, a TF-IDF + Logistic Regression classifier,
evaluation, and a Flask web app + REST API for live predictions.

## Project structure

```
sentiment_analysis_project/
├── README.md
├── requirements.txt
├── app.py                     # Flask web app + REST API
├── data/
│   ├── generate_data.py       # Creates the synthetic review dataset
│   └── reviews.csv            # Generated dataset (~2,150 reviews)
├── src/
│   ├── preprocess.py          # Text cleaning (train + inference share this)
│   ├── train.py                # Trains model, evaluates, saves artifacts
│   └── predict.py              # Loads model, scores new review text
├── model/                     # Created after training
│   ├── sentiment_model.joblib # Trained pipeline (TF-IDF + classifier)
│   ├── metrics.json           # Evaluation metrics
│   ├── confusion_matrix.png
│   └── top_words.png          # Most predictive words per sentiment class
└── templates/
    └── index.html             # Web UI
```

## What it does

Classifies a product review as **positive**, **negative**, or
**neutral**. This is the classic NLP "sentiment analysis" task —
the same shape of problem used for social media monitoring, customer
feedback triage, and review analysis in production systems.

- **Text pipeline**: lowercase → strip URLs/HTML/punctuation → TF-IDF
  vectorization (unigrams + bigrams) → Logistic Regression.
- **Tuning**: `GridSearchCV` over `max_features` and regularization
  strength `C`, scored on macro F1 (fair across all 3 classes).
- **Class imbalance**: handled with `class_weight="balanced"`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset
```bash
python data/generate_data.py
```
A CSV is already included, so this is optional unless you want a
fresh/different sample.

### 2. Train the model
```bash
python src/train.py
```
Prints accuracy, F1, and a full classification report, then saves:
- `model/sentiment_model.joblib` — the trained pipeline
- `model/confusion_matrix.png`
- `model/top_words.png` — the words most associated with each class

### 3. Predict from Python
```bash
python src/predict.py
```
Runs 5 hand-written example reviews (not from the training templates)
through the model. Or import it directly:

```python
from src.predict import predict_sentiment

result = predict_sentiment("This is the best purchase I've made, love it!")
# -> {"sentiment": "positive", "confidence": 0.85, "probabilities": {...}}
```

### 4. Run the web app + API
```bash
python app.py
```
Open **http://127.0.0.1:5001** for a text box UI, or call the API:

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible quality, broke within two days."}'
# -> {"sentiment": "negative", "confidence": 0.97, "probabilities": {...}}
```

`GET /api/health` returns `{"status": "ok"}`.

## A note on accuracy

This model scores ~100% on the held-out synthetic test set. That's
expected, not a bug: the synthetic templates use fairly distinct
vocabulary per class (e.g. "love"/"amazing" vs. "terrible"/"broke"),
so TF-IDF separates them cleanly. On fresh, hand-written sentences
(see `src/predict.py`'s examples) it still generalizes correctly with
strong confidence — but real-world review text is messier (sarcasm,
mixed sentiment, typos), so expect more like 80-90% accuracy on a
real dataset.

To make this production-realistic, swap `data/reviews.csv` for a
real dataset — e.g. the IMDB Movie Reviews or Amazon Product Reviews
datasets on Kaggle/HuggingFace — keeping the same two columns
(`review`, `sentiment`). No other code needs to change.

## Extending this project

- Swap Logistic Regression for a fine-tuned transformer (e.g.
  `distilbert-base-uncased` via HuggingFace) for a large accuracy
  jump on real-world messy text.
- Add a neutral-vs-mixed-sentiment class if your use case needs it.
- Add explainability with `eli5` or LIME to show which words drove
  a given prediction in the UI.
- Batch-score a CSV of reviews via a new `/api/predict_batch` endpoint.
- Containerize with Docker and deploy behind `gunicorn` for production.

## Notes

- The dataset is **synthetically generated** (see
  `data/generate_data.py`) from templates, purely for demonstration.
  Swap in real review data before drawing business conclusions.
