# Model Card — Customer Sentiment and Star Rating

## Models
- 3-class sentiment: Negative / Neutral / Positive.
- 5-class rating: 1, 2, 3, 4, or 5 stars.
- Both models are trained from the same fine-grained Amazon review dataset, but use separate targets.

## Training and evaluation
Run `python train_models.py`. It loads up to 50,000 labeled training rows and 10,000 labeled test rows from `yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes`, trains word + character TF-IDF Logistic Regression models, and writes `model_metrics.json`. Reported metrics should come from that run, not guessed values.

## Confidence
The displayed confidence is the highest class probability returned by Logistic Regression. It is a model score, not a guarantee of correctness and is not a calibrated probability. Treat low scores as uncertain; test on representative reviews before deployment.

## Limitations
- The dataset is mainly English Amazon reviews; Hinglish, Hindi, slang, sarcasm, and domain shifts may perform poorly.
- Star ratings are ordinal. A one-star error (e.g. 4 predicted as 5) is less severe than a four-star error.
- Do not claim improved accuracy until `model_metrics.json` from a completed training run has been reviewed.
