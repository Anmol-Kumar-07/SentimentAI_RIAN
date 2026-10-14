"""Train 3-class sentiment and 1-5 star rating models from the original labeled dataset.
Run once before starting the Flask app. Requires internet to download the Hugging Face dataset.
"""
import re, json, time
from itertools import islice
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / 'data'
DATA_DIR.mkdir(exist_ok=True)
DATASET_ID = 'yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes'
SEED = 42

# Keep negation words; removing all punctuation is not necessary for sentiment.
def clean_text(value):
    text = '' if value is None else str(value).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'[^a-z0-9\s\'!?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

print('Downloading/loading labeled Amazon review dataset...')
# Streaming avoids downloading/building the entire multi-million-row dataset cache.
# We only take a manageable sample for local training.
ds = load_dataset(DATASET_ID, streaming=True)
train_rows = list(islice(ds['train'], 50000))
test_rows = list(islice(ds['test'], 10000))
train_df = pd.DataFrame(train_rows)
test_df = pd.DataFrame(test_rows)
if train_df.empty or test_df.empty:
    raise RuntimeError('Dataset stream returned no rows. Check internet access or dataset availability.')
for df in (train_df, test_df):
    # The published dataset uses class_index 1..5 and review_text.
    if 'class_index' not in df or 'review_text' not in df:
        raise ValueError('Dataset schema changed: expected class_index and review_text columns.')
    df['rating'] = pd.to_numeric(df['class_index'], errors='coerce').astype(int)
    df['clean_text'] = df['review_text'].map(clean_text)
    df.dropna(subset=['rating'], inplace=True)
    df.drop(df[df['clean_text'].str.len() == 0].index, inplace=True)

# The source dataset is rating-labeled, so both targets come from the same true label.
def to_sentiment(stars):
    if stars <= 2: return 'Negative'
    if stars == 3: return 'Neutral'
    return 'Positive'
train_df['sentiment'] = train_df['rating'].map(to_sentiment)
test_df['sentiment'] = test_df['rating'].map(to_sentiment)

# Word + character features help with spelling variation; class_index labels remain separate.
def make_vectorizer():
    return FeatureUnion([
        ('word', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=180000, sublinear_tf=True, strip_accents='unicode')),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=3, max_features=100000, sublinear_tf=True, strip_accents='unicode')),
    ])

X_train = train_df['clean_text'].tolist()
X_test = test_df['clean_text'].tolist()
reports = {}
for target, suffix, labels in [
    ('sentiment', '3class', ['Negative','Neutral','Positive']),
    ('rating', '5class', [1,2,3,4,5]),
]:
    print(f'\nTraining {suffix} model...')
    vectorizer = make_vectorizer()
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    ytr = train_df[target].to_numpy()
    yte = test_df[target].to_numpy()
    model = LogisticRegression(C=4.0, max_iter=1200, class_weight='balanced', solver='lbfgs', random_state=SEED)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    accuracy = float(accuracy_score(yte, pred))
    report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
    reports[suffix] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(yte, pred, labels=labels).tolist(),
        'labels': labels,
        'train_samples': int(len(train_df)), 'test_samples': int(len(test_df)),
        'dataset': DATASET_ID,
        'feature_method': 'word TF-IDF (1-2 grams) + character TF-IDF (3-5 grams)',
        'model': 'LogisticRegression(C=4.0, class_weight=balanced)',
    }
    joblib.dump(model, BASE / f'{"sentiment_3class" if suffix == "3class" else "rating_5class"}_logistic_regression_model.pkl', compress=3)
    joblib.dump(vectorizer, BASE / f'tfidf_{suffix}_vectorizer.pkl', compress=3)
    print(f'{suffix} accuracy: {accuracy:.4f}')
    print(classification_report(yte, pred, labels=labels, zero_division=0))

reports['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
(BASE / 'model_metrics.json').write_text(json.dumps(reports, indent=2, default=str), encoding='utf-8')
print('\nSaved both models, vectorizers and model_metrics.json. Start the Flask app now.')
