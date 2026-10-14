# Power BI integration (Windows)

1. Start the Flask app with `run_windows.bat`.
2. Submit a few sample reviews in the app so SQLite has records.
3. Open `http://127.0.0.1:5000/export/reviews.csv` and save the CSV file.
4. In Power BI Desktop, select **Get data → Text/CSV**, choose `sentiment_reviews_for_powerbi.csv`, then **Load**.
5. Create visuals: count of reviews by `predicted_sentiment`, confidence average by sentiment, and review counts by `created_at` date. Use `id` as a row identifier.
6. Export again and refresh the CSV source after collecting more predictions. The CSV contains predictions submitted to this local app, not all 3.65M source reviews.

## Suggested visuals
- Donut chart: Legend = `predicted_sentiment`, Values = Count of `id`.
- Line chart: X-axis = `created_at`, Y-axis = Count of `id`.
- Bar chart: Axis = `predicted_sentiment`, Values = Average of `confidence`.
- Table: `review_text`, `predicted_sentiment`, `confidence`, `created_at`.
