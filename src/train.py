"""
train.py
--------
Trains a sentiment classifier (TF-IDF + Logistic Regression) on the
review dataset, evaluates it, and saves the fitted pipeline.

Run:
    python src/train.py
"""

import os
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from preprocess import clean_series

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_THIS_DIR, "..")
DATA_PATH = os.path.join(_ROOT, "data", "reviews.csv")
MODEL_PATH = os.path.join(_ROOT, "model", "sentiment_model.joblib")
METRICS_PATH = os.path.join(_ROOT, "model", "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(_ROOT, "model", "confusion_matrix.png")
TOP_WORDS_PATH = os.path.join(_ROOT, "model", "top_words.png")


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["clean_review"] = clean_series(df["review"])

    X = df["clean_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)}  Test size: {len(X_test)}")

    pipeline = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    param_grid = {
        "tfidf__max_features": [2000, 5000],
        "classifier__C": [0.5, 1.0, 3.0],
    }

    print("Running grid search...")
    grid = GridSearchCV(pipeline, param_grid, cv=4, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")

    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "best_params": grid.best_params_,
    }

    print("\n=== Evaluation ===")
    print(f"  accuracy: {metrics['accuracy']:.4f}")
    print(f"  f1_macro: {metrics['f1_macro']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # --- Confusion matrix plot ---
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix — Sentiment Classifier")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nSaved confusion matrix to {CONFUSION_MATRIX_PATH}")

    # --- Top predictive words per class ---
    tfidf = best_model.named_steps["tfidf"]
    clf = best_model.named_steps["classifier"]
    feature_names = tfidf.get_feature_names_out()

    fig, axes = plt.subplots(1, len(clf.classes_), figsize=(5 * len(clf.classes_), 4))
    for i, cls in enumerate(clf.classes_):
        coefs = clf.coef_[i]
        top_idx = coefs.argsort()[-10:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        top_vals = [coefs[j] for j in top_idx]
        ax = axes[i] if len(clf.classes_) > 1 else axes
        sns.barplot(x=top_vals, y=top_words, ax=ax, color="#4C72B0")
        ax.set_title(f"Top words: {cls}")
    plt.tight_layout()
    plt.savefig(TOP_WORDS_PATH, dpi=150)
    print(f"Saved top-words plot to {TOP_WORDS_PATH}")

    # --- Save artifacts ---
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved trained pipeline to {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
