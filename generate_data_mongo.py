import pandas as pd
import pymongo
import json
from datetime import datetime, timedelta
import random

with open('config/db_config.json') as f:
    DB_CONFIG = json.load(f)

# Generate sample MongoDB data
def generate_sample_news_data(num_records=100):
    """Generate sample financial news data for MongoDB with sentiment scores"""
    print(f"Generating {num_records} sample financial news records...")

    news_sources = ["Bloomberg", "CNBC", "Reuters", "Wall Street Journal", "Financial Times"]
    ticker_pool = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "JNJ", "WMT"]

    news_data = []

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    for i in range(num_records):
        # Random date within the last 30 days
        random_days = random.randint(0, 30)
        news_date = end_date - timedelta(days=random_days)

        # Generate 1-3 random tickers for each news item
        num_tickers = random.randint(1, 3)
        tickers = random.sample(ticker_pool, num_tickers)

        # Generate sentiment score (-1.0 to 1.0)
        sentiment_score = round(random.uniform(-1.0, 1.0), 2)

        # Determine sentiment category based on score
        if sentiment_score > 0.2:
            sentiment = "positive"
        elif sentiment_score < -0.2:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Generate news title based on tickers and sentiment
        title_prefix = {
            "positive": ["Bullish Outlook:", "Stocks Rally:", "Upward Trend:"],
            "negative": ["Bearish Signs:", "Stocks Plunge:", "Downward Pressure:"],
            "neutral": ["Market Update:", "Stocks Fluctuate:", "Industry Analysis:"]
        }

        title = f"{random.choice(title_prefix[sentiment])} {', '.join(tickers)} {random.choice(['Analysis', 'Report', 'Forecast', 'Overview'])}"

        news_item = {
            "title": title,
            "source": random.choice(news_sources),
            "published_date": news_date,
            "summary": f"This is a sample summary for financial news about {', '.join(tickers)}.",
            "url": f"https://example.com/financial-news/{i}",
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "tickers": tickers,
            "created_at": datetime.now()
        }

        news_data.append(news_item)

    print(f"Successfully generated {len(news_data)} sample news records")
    return news_data


# Function to load data to MongoDB
def load_to_mongodb(news_data):
    """Load financial news data to MongoDB"""
    print("Loading financial news data to MongoDB...")

    # Connect to MongoDB using the format from your extraction function
    mongo_config = DB_CONFIG['mongodb']
    client = pymongo.MongoClient(mongo_config['host'], mongo_config['port'])
    db = client[mongo_config['db']]
    collection = db['financial_news']

    # Insert data
    if isinstance(news_data, pd.DataFrame):
        # Convert DataFrame to list of dictionaries
        news_data = news_data.to_dict('records')

    # Insert many documents
    result = collection.insert_many(news_data)
    client.close()

    print(f"Successfully loaded {len(result.inserted_ids)} news records to MongoDB")
    return True


# Example usage
if __name__ == "__main__":
    # Generate sample news data
    sample_news_data = generate_sample_news_data(100)

    # Load sample data to MongoDB
    load_to_mongodb(sample_news_data)
