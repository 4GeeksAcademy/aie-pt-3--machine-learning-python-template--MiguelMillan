"""Run the fixed nlptown sentiment model on customer reviews."""

from pathlib import Path

import pandas as pd
from transformers import pipeline


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
MODEL_REVISION = "8f6f4e3a8f70be4b65d3a4a8762b6d781cda240d"
LABEL_TO_STARS = {"1 star": 1, "2 stars": 2, "3 stars": 3, "4 stars": 4, "5 stars": 5}


def load_sentiment_model():
	"""Load the agreed pretrained model once and return the inference pipeline."""
	return pipeline(
		task="sentiment-analysis",
		model=MODEL_NAME,
		tokenizer=MODEL_NAME,
		revision=MODEL_REVISION,
		device=-1,
	)


def predict_reviews(reviews: pd.DataFrame, sentiment_model) -> pd.DataFrame:
	"""Append model star predictions, confidence, and sentiment bands."""
	if "review_text" not in reviews.columns:
		raise ValueError("Input data must contain a 'review_text' column")

	enriched = reviews.copy()
	texts = enriched["review_text"].fillna("").astype(str).tolist()
	results = sentiment_model(texts, batch_size=16, truncation=True)
	labels = [result["label"] for result in results]
	enriched["predicted_stars"] = [LABEL_TO_STARS[label] for label in labels]
	enriched["sentiment_confidence"] = [result["score"] for result in results]
	enriched["sentiment_band"] = enriched["predicted_stars"].map(
		lambda stars: "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"
	)
	return enriched


def run_inference(
	input_path: Path | str | None = None,
	output_path: Path | str | None = None,
) -> pd.DataFrame:
	"""Read the raw CSV, load the model once, and write enriched review rows."""
	project_root = Path(__file__).resolve().parents[1]
	source = Path(input_path) if input_path else project_root / "data/raw/reviews.csv"
	destination = (
		Path(output_path)
		if output_path
		else project_root / "data/processed/reviews_with_sentiment.csv"
	)

	reviews = pd.read_csv(source)
	sentiment_model = load_sentiment_model()
	enriched = predict_reviews(reviews, sentiment_model)
	destination.parent.mkdir(parents=True, exist_ok=True)
	enriched.to_csv(destination, index=False)
	print(f"Wrote {len(enriched)} reviews to {destination}")
	return enriched


if __name__ == "__main__":
	run_inference()
