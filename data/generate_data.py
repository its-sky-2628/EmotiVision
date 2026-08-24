"""
generate_data.py
-----------------
Generates a synthetic product-review dataset for sentiment analysis
(positive / negative / neutral) and saves it to data/reviews.csv

Run:
    python data/generate_data.py
"""

import random
import pandas as pd

random.seed(42)

# --- Building blocks for realistic, varied review sentences ---

POSITIVE_TEMPLATES = [
    "I absolutely love this {product}, it works perfectly!",
    "Great {product}, exceeded my expectations.",
    "This {product} is amazing, worth every penny.",
    "Excellent quality, {product} arrived on time and works great.",
    "Best {product} I've ever bought, highly recommend it.",
    "Superb build quality, this {product} is fantastic.",
    "Very happy with this {product}, five stars!",
    "The {product} works flawlessly, I'm impressed.",
    "Fantastic {product}, customer service was also great.",
    "This {product} exceeded all my expectations, love it.",
    "Perfect {product} for the price, couldn't be happier.",
    "Really good {product}, does exactly what it promises.",
    "Outstanding {product}, will definitely buy again.",
    "I'm so glad I bought this {product}, it's wonderful.",
    "Top notch {product}, fast delivery and great packaging.",
]

NEGATIVE_TEMPLATES = [
    "This {product} is terrible, broke after one day.",
    "Very disappointed with this {product}, waste of money.",
    "The {product} stopped working within a week, awful.",
    "Poor quality {product}, would not recommend.",
    "This {product} is a scam, doesn't work as advertised.",
    "Horrible experience with this {product}, never buying again.",
    "The {product} arrived damaged and customer service was rude.",
    "Worst {product} I've ever purchased, total waste.",
    "This {product} is cheaply made and fell apart quickly.",
    "Extremely disappointed, the {product} does not work at all.",
    "Bad {product}, completely different from what was advertised.",
    "The {product} is defective, requesting a refund immediately.",
    "Terrible build quality, this {product} is a huge letdown.",
    "I regret buying this {product}, it's just useless.",
    "Awful {product}, broke within days of use.",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay, nothing special about it.",
    "Average {product}, does the job but nothing more.",
    "This {product} is fine, could be better though.",
    "The {product} works as expected, no complaints really.",
    "It's an average {product}, meets basic requirements.",
    "The {product} is decent for the price, not amazing.",
    "Not bad, not great, just a regular {product}.",
    "This {product} does what it says, nothing extraordinary.",
    "The {product} is functional but the design could improve.",
    "Reasonably priced {product}, performs adequately.",
    "The {product} is okay for casual use, nothing outstanding.",
    "It's a standard {product}, works fine for basic needs.",
]

PRODUCTS = [
    "phone", "laptop", "headphones", "blender", "backpack", "watch",
    "camera", "keyboard", "mouse", "speaker", "charger", "monitor",
    "tablet", "vacuum cleaner", "microwave", "shoes", "jacket",
    "book", "coffee maker", "router", "printer", "television",
]

# Extra free-form snippets to append sometimes for more variety
POSITIVE_SUFFIXES = [
    " Shipping was fast too.",
    " Definitely a great buy.",
    " My whole family loves it.",
    " Will be buying more from this brand.",
    "",
    "",
]

NEGATIVE_SUFFIXES = [
    " Asked for a refund.",
    " Do not buy this.",
    " Complete waste of money.",
    " Never ordering from here again.",
    "",
    "",
]

NEUTRAL_SUFFIXES = [
    " Might consider other brands next time.",
    " It's okay for the price.",
    " Nothing to write home about.",
    "",
    "",
]


def make_review(templates, suffixes, product):
    template = random.choice(templates)
    suffix = random.choice(suffixes)
    review = template.format(product=product) + suffix
    # Randomly drop/shorten to add noise, like real reviews vary in length
    if random.random() < 0.15:
        words = review.split()
        cut = max(3, int(len(words) * random.uniform(0.5, 0.8)))
        review = " ".join(words[:cut])
    return review


# Short, ambiguous, harder-to-classify reviews mixed in for realism
SHORT_POSITIVE = ["Good {product}.", "Love it.", "Works well.", "Pretty good, satisfied.", "Nice {product}, thanks."]
SHORT_NEGATIVE = ["Bad {product}.", "Hate it.", "Doesn't work.", "Not satisfied at all.", "Waste, avoid this {product}."]
SHORT_NEUTRAL = ["It's fine.", "Okay {product}.", "Meh, average.", "So-so experience.", "Neither good nor bad."]


def generate_dataset(n_per_class=1200):
    rows = []
    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        if random.random() < 0.25:
            text = random.choice(SHORT_POSITIVE).format(product=product)
        else:
            text = make_review(POSITIVE_TEMPLATES, POSITIVE_SUFFIXES, product)
        rows.append({"review": text, "sentiment": "positive"})
    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        if random.random() < 0.25:
            text = random.choice(SHORT_NEGATIVE).format(product=product)
        else:
            text = make_review(NEGATIVE_TEMPLATES, NEGATIVE_SUFFIXES, product)
        rows.append({"review": text, "sentiment": "negative"})
    for _ in range(n_per_class):
        product = random.choice(PRODUCTS)
        if random.random() < 0.25:
            text = random.choice(SHORT_NEUTRAL).format(product=product)
        else:
            text = make_review(NEUTRAL_TEMPLATES, NEUTRAL_SUFFIXES, product)
        rows.append({"review": text, "sentiment": "neutral"})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df = df.drop_duplicates(subset="review").reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "data/reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["sentiment"].value_counts())
