"""
preprocess.py
-------------
Text cleaning utilities used by both train.py and predict.py so
training and inference always apply identical preprocessing.
"""

import re
import string


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/extra whitespace."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                       # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                     # keep only letters
    text = re.sub(r"\s+", " ", text).strip()                 # collapse whitespace
    return text


def clean_series(series):
    """Apply clean_text to a pandas Series of raw text."""
    return series.astype(str).apply(clean_text)
