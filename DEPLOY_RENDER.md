# Deploy SentimentAI RIAN on Render

1. Extract this ZIP. Create a GitHub repository and upload the *contents* of this project folder (not the outer folder).
2. In Render, choose **New + → Web Service**, connect the repository, and use the settings from `render.yaml` (or let Render detect it).
3. Add `YOUTUBE_API_KEY` under Render → Environment if you want YouTube comment analysis. Never commit the real API key.
4. Deploy and test `/health`, the homepage, prediction, and `/youtube`.

## Important notes
- This service uses SQLite at `sentiment.db`. On hosts without a persistent disk, database records may be lost when the service is redeployed or its instance is replaced. Add a persistent disk and move the database path onto it, or use a managed database, if review history must persist.
- The project contains pre-trained model files. Runtime installation should not need to retrain them.
- The free hosting tier may sleep when idle and can have cold-start delays.
