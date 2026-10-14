from flask import Flask, render_template, request, jsonify, Response
from pathlib import Path
from datetime import datetime
import csv, io, json, logging, sqlite3, re
from logging.handlers import RotatingFileHandler
import joblib

BASE = Path(__file__).resolve().parent
SENTIMENT_MODEL_PATH = BASE / 'sentiment_3class_logistic_regression_model.pkl'
SENTIMENT_VEC_PATH = BASE / 'tfidf_3class_vectorizer.pkl'
RATING_MODEL_PATH = BASE / 'rating_5class_logistic_regression_model.pkl'
RATING_VEC_PATH = BASE / 'tfidf_5class_vectorizer.pkl'
METRICS_PATH = BASE / 'model_metrics.json'
DB_PATH = BASE / 'sentiment.db'
LOG_DIR = BASE / 'logs'
LOG_DIR.mkdir(exist_ok=True)
app = Flask(__name__)
handler = RotatingFileHandler(LOG_DIR / 'app.log', maxBytes=500_000, backupCount=3, encoding='utf-8')
handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

missing = [p.name for p in [SENTIMENT_MODEL_PATH, SENTIMENT_VEC_PATH] if not p.exists()]
if missing:
    raise RuntimeError('Missing trained model file(s): ' + ', '.join(missing) + '. Run: python train_models.py')
sentiment_model = joblib.load(SENTIMENT_MODEL_PATH)
sentiment_vectorizer = joblib.load(SENTIMENT_VEC_PATH)
rating_model = joblib.load(RATING_MODEL_PATH) if RATING_MODEL_PATH.exists() else None
rating_vectorizer = joblib.load(RATING_VEC_PATH) if RATING_VEC_PATH.exists() else None

try:
    METRICS = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
except Exception:
    METRICS = {}
SENTIMENT_METRICS = METRICS.get('3class', {})
RATING_METRICS = METRICS.get('5class', {})
MODEL_VERSION = '2.0.0'
MODEL_INFO = {
    'model_name': 'Word + character TF-IDF with Logistic Regression',
    'model_version': MODEL_VERSION,
    'trained_dataset': 'yassiracharki/Amazon_Reviews_for_Sentiment_Analysis_fine_grained_5_classes',
    'sentiment_accuracy': SENTIMENT_METRICS.get('accuracy'),
    'rating_accuracy': RATING_METRICS.get('accuracy') if rating_model is not None else None,
    'samples': SENTIMENT_METRICS.get('train_samples'),
    'labels': ['Negative', 'Neutral', 'Positive'],
    'rating_labels': [1, 2, 3, 4, 5],
    'note': 'Metrics are loaded from the most recent training run in model_metrics.json.'
}



def extract_youtube_video_id(url):
    """Accept common YouTube watch, short, embed and live URLs."""
    value = (url or '').strip()
    patterns = [
        r'(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})',
        r'(?:youtu\.be/)([A-Za-z0-9_-]{11})',
        r'youtube\.com/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    raise ValueError('Enter a valid YouTube video URL (watch, youtu.be, shorts, or live link).')


def _youtube_api_key():
    """Read API key from environment or a local .env file without extra packages."""
    import os
    key = os.environ.get('YOUTUBE_API_KEY', '').strip()
    if key:
        return key
    env_file = BASE / '.env'
    if env_file.exists():
        for line in env_file.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('YOUTUBE_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    return ''


def analyze_youtube_comments(video_url, max_comments=200):
    """Fetch public top-level comments through the official YouTube Data API v3."""
    import urllib.parse
    import urllib.request
    import urllib.error

    video_id = extract_youtube_video_id(video_url)
    canonical_url = f'https://www.youtube.com/watch?v={video_id}'
    api_key = _youtube_api_key()
    if not api_key:
        raise RuntimeError(
            'YouTube Data API key is not configured. Create an API key with YouTube Data API v3 enabled, '
            'then add YOUTUBE_API_KEY=YOUR_API_KEY to the project .env file and restart the app.'
        )

    results = []
    page_token = None
    while len(results) < max_comments:
        params = {
            'part': 'snippet', 'videoId': video_id,
            'maxResults': min(100, max_comments - len(results)),
            'textFormat': 'plainText', 'order': 'relevance', 'key': api_key,
        }
        if page_token:
            params['pageToken'] = page_token
        api_url = 'https://www.googleapis.com/youtube/v3/commentThreads?' + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(api_url, headers={'Accept': 'application/json', 'User-Agent': 'CustomerSentimentAnalysis/1.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                payload = json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode('utf-8')).get('error', {}).get('errors', [{}])[0].get('reason', '')
            except Exception:
                detail = ''
            if detail == 'commentsDisabled':
                raise RuntimeError('Comments are disabled for this video.') from exc
            if detail in ('quotaExceeded', 'dailyLimitExceeded'):
                raise RuntimeError('YouTube Data API quota has been exceeded. Try again later or check your Google Cloud quota.') from exc
            if exc.code in (400, 403):
                raise RuntimeError('YouTube API rejected the request. Check that YouTube Data API v3 is enabled and the API key is valid.') from exc
            if exc.code == 404:
                raise RuntimeError('The video was not found or is not publicly accessible.') from exc
            raise RuntimeError(f'YouTube API request failed (HTTP {exc.code}).') from exc
        except Exception as exc:
            raise RuntimeError(f'Unable to connect to the YouTube Data API: {exc}') from exc

        for item in payload.get('items', []):
            snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
            text = str(snippet.get('textDisplay', '')).strip()
            if not text:
                continue
            review, sentiment, sent_conf, rating, rating_conf = predict(text)
            results.append({
                'video_url': canonical_url,
                'author': snippet.get('authorDisplayName', ''),
                'comment_text': review,
                'predicted_sentiment': sentiment,
                'sentiment_confidence': round(sent_conf, 4),
                'predicted_rating': rating,
                'rating_confidence': round(rating_conf, 4),
                'likes': snippet.get('likeCount', 0),
                'published_time': snippet.get('publishedAt', ''),
                'model_version': MODEL_VERSION,
            })
            if len(results) >= max_comments:
                break
        page_token = payload.get('nextPageToken')
        if not page_token or not payload.get('items'):
            break

    if not results:
        raise RuntimeError('The API returned no public comments for this video. Comments may be disabled or unavailable.')
    return canonical_url, results


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        # Keep both the legacy `confidence` column and the newer explicit
        # sentiment_confidence column so older and newer SQLite databases work.
        conn.execute('''CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            sentiment_confidence REAL NOT NULL DEFAULT 0,
            predicted_rating INTEGER NOT NULL DEFAULT 3,
            rating_confidence REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT '',
            model_version TEXT NOT NULL DEFAULT '2.0.0'
        )''')
        cols = {r[1] for r in conn.execute('PRAGMA table_info(reviews)')}
        migrations = [
            ('review_text', "TEXT NOT NULL DEFAULT ''"),
            ('predicted_sentiment', "TEXT NOT NULL DEFAULT 'Neutral'"),
            ('confidence', 'REAL NOT NULL DEFAULT 0'),
            ('sentiment_confidence', 'REAL NOT NULL DEFAULT 0'),
            ('predicted_rating', 'INTEGER NOT NULL DEFAULT 3'),
            ('rating_confidence', 'REAL NOT NULL DEFAULT 0'),
            ('created_at', "TEXT NOT NULL DEFAULT ''"),
            ('model_version', "TEXT NOT NULL DEFAULT '2.0.0'")]
        for name, declaration in migrations:
            if name not in cols:
                conn.execute(f'ALTER TABLE reviews ADD COLUMN {name} {declaration}')
        # Preserve historical values when migrating older database layouts.
        cols = {r[1] for r in conn.execute('PRAGMA table_info(reviews)')}
        if 'confidence' in cols:
            conn.execute('UPDATE reviews SET sentiment_confidence=confidence WHERE sentiment_confidence=0 AND confidence<>0')
        conn.execute('UPDATE reviews SET confidence=sentiment_confidence WHERE confidence=0 AND sentiment_confidence<>0')

def predict(review):
    review = '' if review is None else str(review).strip()
    if not review:
        raise ValueError('Please enter a review.')
    sx = sentiment_vectorizer.transform([review])
    sp = sentiment_model.predict_proba(sx)[0]
    sc = sentiment_model.classes_[int(sp.argmax())]
    # New training stores string labels; handle older numeric artifacts defensively.
    if isinstance(sc, (int,)) or str(sc).isdigit():
        sc = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}.get(int(sc), 'Neutral')
    else:
        sc = str(sc).title()
    sentiment_confidence = float(sp.max())
    if rating_model is not None and rating_vectorizer is not None:
        rx = rating_vectorizer.transform([review])
        rp = rating_model.predict_proba(rx)[0]
        rating = int(rating_model.classes_[int(rp.argmax())])
        rating_confidence = float(rp.max())
    else:
        # Fallback when only the user's two 3-class sentiment artifacts are present.
        # This is a sentiment-derived estimate, not a separately trained rating model.
        rating = {'Negative': 2, 'Neutral': 3, 'Positive': 4}.get(sc, 3)
        rating_confidence = sentiment_confidence
    return review, sc, sentiment_confidence, rating, rating_confidence

def save_review(review, sentiment, sent_conf, rating, rating_conf):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        # Backward-compatible insert: older sentiment.db schemas require the legacy
        # NOT NULL `confidence` column. Write it alongside the newer confidence field.
        conn.execute('''INSERT INTO reviews(review_text,predicted_sentiment,confidence,sentiment_confidence,predicted_rating,rating_confidence,created_at,model_version)
            VALUES(?,?,?,?,?,?,?,?)''', (review, sentiment, sent_conf, sent_conf, rating, rating_conf,
            datetime.now().astimezone().isoformat(timespec='seconds'), MODEL_VERSION))

@app.route('/')
def index():
    return render_template('index.html', metrics=MODEL_INFO, error=None)

@app.route('/predict', methods=['POST'])
def web_predict():
    try:
        review, sentiment, sent_conf, rating, rating_conf = predict(request.form.get('review', ''))
        save_review(review, sentiment, sent_conf, rating, rating_conf)
        app.logger.info('prediction_saved sentiment=%s rating=%s', sentiment, rating)
        return render_template('result.html', review=review, sentiment=sentiment,
            confidence=round(sent_conf*100, 1), rating=rating, rating_confidence=round(rating_conf*100, 1))
    except ValueError as exc:
        return render_template('index.html', metrics=MODEL_INFO, error=str(exc)), 400
    except Exception:
        app.logger.exception('web prediction failed')
        return render_template('index.html', metrics=MODEL_INFO, error='Prediction failed. Check logs/app.log.'), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(silent=True) or {}
    try:
        review, sentiment, sent_conf, rating, rating_conf = predict(data.get('review', ''))
        save_review(review, sentiment, sent_conf, rating, rating_conf)
        return jsonify({'review': review, 'sentiment': sentiment, 'sentiment_confidence': round(sent_conf,4),
            'predicted_rating': rating, 'rating_confidence': round(rating_conf,4), 'model_version': MODEL_VERSION})
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/youtube', methods=['GET', 'POST'])
def youtube_analyzer():
    error = None
    results = []
    summary = {}
    video_url = ''
    requested_count = 200
    if request.method == 'POST':
        video_url = request.form.get('video_url', '').strip()
        try:
            requested_count = int(request.form.get('max_comments', '200'))
            requested_count = max(1, min(requested_count, 1000))
            canonical_url, results = analyze_youtube_comments(video_url, requested_count)
            video_url = canonical_url
            for label in ('Positive', 'Negative', 'Neutral'):
                summary[label] = sum(1 for row in results if row['predicted_sentiment'] == label)
            total = len(results)
            summary = {label: {'count': count, 'percent': round(count * 100 / total, 1)} for label, count in summary.items()}
            # Keep each analyzed comment in the same review-history database used by the app.
            for row in results:
                try:
                    save_review(row['comment_text'], row['predicted_sentiment'], row['sentiment_confidence'],
                                row['predicted_rating'], row['rating_confidence'])
                except Exception:
                    app.logger.exception('Could not save a YouTube comment to review history')
            app.logger.info('youtube_analysis_done comments=%d video=%s', total, video_url)
        except (ValueError, RuntimeError) as exc:
            error = str(exc)
        except Exception:
            app.logger.exception('YouTube analysis failed')
            error = 'YouTube analysis failed. Check logs/app.log for technical details.'
    return render_template('youtube.html', error=error, results=results, summary=summary,
                           video_url=video_url, requested_count=requested_count)


@app.route('/youtube/export', methods=['POST'])
def export_youtube_csv():
    rows = request.form.get('results_json', '')
    try:
        records = json.loads(rows)
        if not isinstance(records, list) or not records:
            raise ValueError('No results')
    except Exception:
        return 'No analyzed comments available for export. Analyze a video first.', 400
    out = io.StringIO(newline='')
    fields = ['id', 'review_text', 'predicted_sentiment', 'sentiment_confidence', 'predicted_rating',
              'rating_confidence', 'created_at', 'model_version', 'video_url', 'author', 'likes', 'published_time']
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    now = datetime.now().astimezone().isoformat(timespec='seconds')
    for i, row in enumerate(records, 1):
        writer.writerow({
            'id': i, 'review_text': row.get('comment_text', ''),
            'predicted_sentiment': row.get('predicted_sentiment', ''),
            'sentiment_confidence': row.get('sentiment_confidence', ''),
            'predicted_rating': row.get('predicted_rating', ''),
            'rating_confidence': row.get('rating_confidence', ''),
            'created_at': now, 'model_version': row.get('model_version', MODEL_VERSION),
            'video_url': row.get('video_url', ''), 'author': row.get('author', ''),
            'likes': row.get('likes', 0), 'published_time': row.get('published_time', '')
        })
    return Response(out.getvalue(), mimetype='text/csv', headers={
        'Content-Disposition': 'attachment; filename=youtube_comment_sentiment_analysis.csv'})

@app.route('/dashboard')
def dashboard():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute('''SELECT id,review_text,predicted_sentiment,sentiment_confidence,predicted_rating,rating_confidence,created_at,model_version
            FROM reviews ORDER BY id DESC LIMIT 200''').fetchall()
        counts = dict(conn.execute('SELECT predicted_sentiment,COUNT(*) FROM reviews GROUP BY predicted_sentiment').fetchall())
        rating_counts = dict(conn.execute('SELECT predicted_rating,COUNT(*) FROM reviews GROUP BY predicted_rating').fetchall())
        daily = conn.execute('SELECT substr(created_at,1,10) day, COUNT(*) total FROM reviews GROUP BY day ORDER BY day DESC LIMIT 14').fetchall()
    total = sum(counts.values())
    for label in ('Negative','Positive','Neutral'): counts.setdefault(label,0)
    for rating in range(1,6): rating_counts.setdefault(rating,0)
    return render_template('dashboard.html', rows=rows, counts=counts, rating_counts=rating_counts, total=total, daily=list(reversed(daily)))

@app.route('/export/reviews.csv')
def export_reviews():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute('''SELECT id,review_text,predicted_sentiment,sentiment_confidence,predicted_rating,rating_confidence,created_at,model_version
            FROM reviews ORDER BY id''').fetchall()
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(['id','review_text','predicted_sentiment','sentiment_confidence','predicted_rating','rating_confidence','created_at','model_version'])
    writer.writerows(rows)
    return Response(out.getvalue(), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=sentiment_reviews_for_powerbi.csv'})

@app.route('/model-info')
def model_info(): return jsonify(MODEL_INFO)

@app.route('/health')
def health():
    return jsonify({'status':'ok','sentiment_model_loaded':SENTIMENT_MODEL_PATH.exists(), 'rating_model_loaded':rating_model is not None, 'rating_mode':'trained_model' if rating_model is not None else 'sentiment-derived estimate', 'database':DB_PATH.name,'model_version':MODEL_VERSION})

if __name__ == '__main__':
    init_db()
    print('Sentiment model loaded.' if rating_model is None else 'Sentiment + 1-5 star rating models loaded.')
    if rating_model is None:
        print('Note: no rating model files found; star rating is a sentiment-derived estimate, not a separately trained prediction.')
    print('App:       http://127.0.0.1:5000')
    print('Dashboard: http://127.0.0.1:5000/dashboard')
    print('YouTube:   http://127.0.0.1:5000/youtube')
    print('Power BI:  http://127.0.0.1:5000/export/reviews.csv')
    app.run(debug=False, host='127.0.0.1', port=5000)
