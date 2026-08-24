"""
predict.py
----------
Loads the trained sentiment pipeline and classifies new review text.

Run as a demo:
    python src/predict.py
"""

import os
import joblib
from preprocess import clean_text

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "..", "model", "sentiment_model.joblib")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_sentiment(text: str) -> dict:
    """
    text: raw review string
    Returns: {"sentiment": "positive"/"negative"/"neutral", "confidence": float,
              "probabilities": {class: prob, ...}}
    """
    model = get_model()
    cleaned = clean_text(text)
    pred = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = model.named_steps["classifier"].classes_
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    confidence = max(prob_dict.values())
    return {"sentiment": pred, "confidence": confidence, "probabilities": prob_dict}


if __name__ == "__main__":
    examples = [
        "This product is absolutely amazing, I love it so much!",
        "Terrible quality, broke within two days, total waste of money.",
        "It's okay, does the job but nothing special.",
        "Best purchase I've made this year, works perfectly!",
        "Really disappointed, would not recommend this to anyone.",
    ]
    for text in examples:
        result = predict_sentiment(text)
        print(f"Review: {text}")
        print(f"  -> {result}\n")
