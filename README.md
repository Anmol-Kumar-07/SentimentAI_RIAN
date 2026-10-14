# SentimentAI RIAN — Customer Sentiment Analysis Capstone — Improved

This project supports **both**:
1. Sentiment classification: Negative / Neutral / Positive.
2. Star-rating prediction: 1–5 stars.

## Windows setup
1. Extract the ZIP.
2. Double-click `run_windows.bat` while connected to the internet. First run creates a virtual environment, installs packages, downloads labeled review data and trains both models. This can take several minutes.
3. When training finishes, Flask starts at `http://127.0.0.1:5000`.
4. Dashboard: `http://127.0.0.1:5000/dashboard`.
5. Power BI CSV: `http://127.0.0.1:5000/export/reviews.csv`.

If you already have the older ZIP, extract this updated ZIP into a **new folder** rather than mixing files.

## Manual setup
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python train_models.py
python app.py
```

## Evaluation
`train_models.py` evaluates on the dataset's separate test split (up to 10,000 records) and saves accuracy, per-class precision/recall/F1, and confusion matrices in `model_metrics.json`. Accuracy is not promised in advance; review the actual test metrics after training.

## API
`POST /api/predict` with JSON `{"review":"The product is excellent"}` returns sentiment, sentiment confidence, predicted stars, rating confidence, and model version.

## Important
The 1–5 star model is trained from actual star labels, not guessed from the sentiment class. The supplied older CSV contains predictions but no true star labels, so it cannot be used by itself to train/evaluate the rating model.


### Windows training fix
The training script now uses Hugging Face streaming and samples up to 50,000 training rows and 10,000 test rows. This avoids building the full multi-million-row dataset cache, which can fail on Windows with a missing `.incomplete` Arrow cache path. Training still requires internet on the first run. Accuracy is measured on the held-out sample and is not guaranteed to reach a particular score.


## YouTube comment sentiment feature

1. Install dependencies using `pip install -r requirements.txt` (the YouTube comment downloader package is included).
2. Start the app with `run_windows.bat`.
3. Open `http://127.0.0.1:5000/youtube`, paste a public YouTube video URL, choose a comment limit, and select **Analyze YouTube comments**.
4. Review the Positive / Negative / Neutral counts and percentages, then download the labeled CSV for Excel or Power BI.

The CSV retains the existing review-analysis fields and adds YouTube metadata (`video_url`, `author`, `likes`, `published_time`). It analyzes accessible public comments only; comments may be unavailable for some videos due to YouTube restrictions, disabled comments, or rate limits.


### Model-file note

This package can run with the two available sentiment artifacts (`sentiment_3class_logistic_regression_model.pkl` and `tfidf_3class_vectorizer.pkl`). If separate 5-class rating model files are not present, the displayed 1–5 rating is a rough sentiment-derived estimate (Negative=2, Neutral=3, Positive=4), not a separately trained rating prediction.


## YouTube Comment Analysis Setup

The YouTube analyzer uses the official YouTube Data API v3. A Google API key is required; public video URLs alone do not guarantee comment access.

1. In Google Cloud Console, create/select a project and enable **YouTube Data API v3**.
2. Create an API key and restrict it appropriately for your use case.
3. In this project folder, create a text file named `.env` (not `.env.txt`).
4. Add one line: `YOUTUBE_API_KEY=YOUR_API_KEY`
5. Restart `run_windows.bat` and open `http://127.0.0.1:5000/youtube`.

The application uses only publicly returned top-level comments. The API may return an error if comments are disabled, the key is invalid, the API is not enabled, or quota is exhausted. All user-facing YouTube error messages are in English. Do not share or commit your API key.


## Recent fixes
- Updated the interface with a brighter SentimentAI RIAN visual identity.
- Fixed SQLite schema migration to support both legacy `confidence` and current `sentiment_confidence` columns, preventing prediction saves from failing on older or newer database files.
- Preserved existing review history during schema upgrades.
