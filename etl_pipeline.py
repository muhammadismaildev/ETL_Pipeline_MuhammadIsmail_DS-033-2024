import numpy as np
import pandas as pd
import requests
import json
import psycopg2
import pymongo
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from oauth2client.service_account import ServiceAccountCredentials
import gspread

# 1. EXTRACTION FUNCTIONS

with open('config/api_keys.json') as f:
    API_KEYS = json.load(f)

with open('config/db_config.json') as f:
    DB_CONFIG = json.load(f)


def pg_engine():
    pg_config = DB_CONFIG['postgres']
    connection_url = f"postgresql://{pg_config['user']}:{pg_config['password']}@{pg_config['host']}:{pg_config['port']}/{pg_config['database']}"
    engine = create_engine(connection_url)
    return engine


def extract_stock_data_csv(file_path):
    """Extract stock market data from CSV files"""
    print("Extracting stock data from CSV...")

    # Read CSV file
    df = pd.read_csv(file_path)

    # Basic validation
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV is missing required columns. Expected: {required_columns}")

    print(f"Successfully extracted {len(df)} stock records")
    return df


def extract_crypto_data_api():
    """Extract cryptocurrency data from CoinGecko API"""
    print("Extracting cryptocurrency data from API...")

    # Set up API endpoint for top 100 cryptocurrencies
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 100,
        'page': 1,
        'sparkline': False,
        'x_cg_api_key': API_KEYS['coingecko']
    }

    # Make API request
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed with status code {response.status_code}")

    # Convert to DataFrame
    data = response.json()
    df = pd.DataFrame(data)
    print(f"Successfully extracted {len(df)} cryptocurrency records")
    return df


def extract_economic_data_sql():
    """Extract economic indicators from PostgreSQL database"""
    print("Extracting economic data from PostgreSQL...")

    # Connect to PostgreSQL
    engine = pg_engine()

    # Query economic indicators
    query = text("""
           SELECT 
               indicator_date, 
               indicator_name, 
               indicator_value,
               region,
               frequency
           FROM economic_indicators
       """)

    # Execute query and fetch data
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"Successfully extracted {len(df)} economic indicator records")
    return df


def extract_news_data_mongodb():
    """Extract financial news sentiment from MongoDB"""
    print("Extracting financial news data from MongoDB...")

    # Connect to MongoDB
    client = pymongo.MongoClient(DB_CONFIG['mongodb']['host'], DB_CONFIG['mongodb']['port'])
    db = client[DB_CONFIG['mongodb']['db']]
    collection = db['financial_news']

    # Query recent news (last 7 days)
    start_date = datetime.now() - timedelta(days=7)

    # Fetch data
    cursor = collection.find({
        'published_date': {'$gte': start_date},
        'sentiment_score': {'$exists': True}
    })

    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    client.close()

    print(f"Successfully extracted {len(df)} financial news records")
    return df


def extract_financial_reports_gdrive():
    """Extract company financial reports from Google Drive"""
    print("Extracting financial reports from Google Drive...")

    GOOGLE_SHEET_NAME = 'financial_reports'

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('gspread-creds.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    print(f"Successfully extracted {len(df)} financial report records")
    return df

# 2. TRANSFORMATION FUNCTIONS

def clean_stock_data(df):
    """Clean and prepare stock market data"""
    print("Cleaning stock market data...")

    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['Date', 'Symbol'])

    # Handle missing values
    df = df.dropna(subset=['Open', 'Close'])  # Remove rows with missing critical values

    # Fill other missing values with appropriate methods
    df['Volume'] = df['Volume'].fillna(0)

    # Ensure numeric data types
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def clean_crypto_data(df):
    """Clean and prepare cryptocurrency data"""
    print("Cleaning cryptocurrency data...")

    # Convert timestamps to datetime
    if 'last_updated' in df.columns:
        start_date = pd.to_datetime('2025-03-24')
        end_date = pd.to_datetime('2025-04-04')
        delta = (end_date - start_date).days
        # df['last_updated'] = pd.to_datetime(df['last_updated']).dt.strftime('%Y-%m-%d')
        # df['last_updated'] = pd.to_datetime(df['last_updated'])

        # Generate a random number of days to add to start_date for each row
        df['last_updated'] = start_date + pd.to_timedelta(np.random.randint(0, delta + 1, size=len(df)), unit='d')

        # Optional: Format the date column as string in 'YYYY-MM-DD' format:
        df['last_updated'] = df['last_updated'].dt.strftime('%Y-%m-%d')

    # Remove duplicates
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id', 'last_updated'])

    # Handle missing or invalid values
    price_cols = [col for col in df.columns if 'price' in col or 'value' in col or 'market_cap' in col]
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=[col])  # Drop rows with missing prices

    return df


def clean_economic_data(df):
    """Clean and prepare economic indicator data"""
    print("Cleaning economic data...")

    # Convert date to datetime
    df['indicator_date'] = pd.to_datetime(df['indicator_date'])

    # Remove duplicates
    df = df.drop_duplicates(subset=['indicator_date', 'indicator_name', 'region'])

    # Handle missing values - for economic data, we'll use forward fill for time series
    df = df.sort_values(['indicator_name', 'region', 'indicator_date'])
    df['indicator_value'] = pd.to_numeric(df['indicator_value'], errors='coerce')
    df['indicator_value'] = df.groupby(['indicator_name', 'region'])['indicator_value'].ffill()

    return df


def clean_news_data(df):
    """Clean and prepare financial news data"""
    print("Cleaning news data...")

    # Convert date to datetime
    if 'published_date' in df.columns:
        df['published_date'] = pd.to_datetime(df['published_date']).dt.strftime('%Y-%m-%d')

    # Remove duplicates (based on title and date)
    if 'title' in df.columns and 'published_date' in df.columns:
        df = df.drop_duplicates(subset=['title', 'published_date'])

    # Ensure sentiment scores are numeric
    if 'sentiment_score' in df.columns:
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')

    # Drop rows with missing sentiment
    df = df.dropna(subset=['sentiment_score'])

    return df


def clean_financial_reports(df):
    """Clean and prepare financial report data"""
    print("Cleaning financial report data...")

    # Handle different date formats from spreadsheets
    if 'report_date' in df.columns:
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')

    # Convert financial metrics to numeric
    financial_cols = [col for col in df.columns if any(term in col.lower()
                                                       for term in
                                                       ['revenue', 'income', 'profit', 'sales', 'assets', 'debt'])]

    for col in financial_cols:
        # Remove currency symbols and commas
        if df[col].dtype == 'object':
            df[col] = df[col].replace({r'\$': '', ',': ''}, regex=True)

        # Convert to float
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with all missing financial values
    df = df.dropna(subset=financial_cols, how='all')

    return df


def standardize_data(stock_df, crypto_df, economic_df, news_df, financial_reports_df):
    """Standardize all datasets to common formats"""
    print("Standardizing data across all sources...")

    # 1. Timestamp standardization (convert all to UTC)
    date_cols = {
        'stock_df': 'Date',
        'crypto_df': 'last_updated',
        'economic_df': 'indicator_date',
        'news_df': 'published_date',
        'financial_reports_df': 'report_date'
    }

    for df_name, date_col in date_cols.items():
        df = locals()[df_name]
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)

            # Rename to standard date column
            df.rename(columns={date_col: 'date'}, inplace=True)

    # 2. Currency standardization (convert all to USD)
    # Assume crypto is already in USD from API
    # For financial reports, we need currency conversion if not in USD
    if 'currency' in financial_reports_df.columns:
        # Example conversion - would need actual exchange rates in production
        exchange_rates = {'EUR': 1.1, 'GBP': 1.3, 'JPY': 0.009}

        # Financial columns to convert
        financial_cols = [col for col in financial_reports_df.columns
                          if any(term in col.lower() for term in
                                 ['revenue', 'income', 'profit', 'sales', 'assets', 'debt'])]

        for currency, rate in exchange_rates.items():
            mask = financial_reports_df['currency'] == currency
            for col in financial_cols:
                financial_reports_df.loc[mask, col] = financial_reports_df.loc[mask, col] * rate

        # Update currency to USD
        financial_reports_df['currency'] = 'USD'

    # 3. Categorize data types
    # For stocks - add asset_type column
    stock_df['asset_type'] = 'Stocks'

    # For crypto - standardize asset type
    crypto_df['asset_type'] = 'Cryptocurrencies'

    # Add data source columns to all dataframes
    stock_df['data_source'] = 'CSV'
    crypto_df['data_source'] = 'API'
    economic_df['data_source'] = 'SQL'
    news_df['data_source'] = 'MongoDB'
    financial_reports_df['data_source'] = 'Google Drive'

    return stock_df, crypto_df, economic_df, news_df, financial_reports_df


def engineer_features(stock_df, crypto_df, economic_df, news_df, financial_reports_df):
    """Engineer additional features from the data"""
    print("Engineering features...")

    # 1. Stock data features
    if not stock_df.empty and all(col in stock_df.columns for col in ['Open', 'Close']):
        # Calculate daily returns
        stock_df['daily_return'] = stock_df.groupby('Symbol')['Close'].pct_change().ffill().fillna(0)

        # Calculate 5-day moving average
        stock_df['ma_5d'] = stock_df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())

        # Calculate volatility (5-day rolling std of returns)
        stock_df['volatility_5d'] = stock_df.groupby('Symbol')['daily_return'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std(ddof=0))

    # 2. Crypto data features
    if not crypto_df.empty and 'current_price' in crypto_df.columns:
        # For demonstration, let's calculate market cap to price ratio
        if all(col in crypto_df.columns for col in ['current_price', 'market_cap']):
            crypto_df['market_cap_to_price'] = crypto_df['market_cap'] / crypto_df['current_price']

    # 3. News sentiment features by entity
    if not news_df.empty and all(col in news_df.columns for col in ['entity', 'sentiment_score']):
        # Calculate rolling 3-day average sentiment by entity
        news_df = news_df.sort_values(['entity', 'date'])
        news_df['sentiment_3d_avg'] = news_df.groupby('entity')['sentiment_score'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean())

    # 4. Financial report metrics
    if not financial_reports_df.empty:
        # Calculate profit margin if revenue and net_income exist
        if all(col in financial_reports_df.columns for col in ['revenue', 'net_income']):
            financial_reports_df['profit_margin'] = financial_reports_df['net_income'] / financial_reports_df['revenue']

        # Calculate debt-to-equity ratio if applicable
        if all(col in financial_reports_df.columns for col in ['total_debt', 'total_equity']):
            financial_reports_df['debt_to_equity'] = financial_reports_df['total_debt'] / financial_reports_df[
                'total_equity']

    return stock_df, crypto_df, economic_df, news_df, financial_reports_df


def aggregate_data(stock_df, crypto_df, economic_df, news_df, financial_reports_df):
    """Aggregate data to create unified market insights"""
    print("Aggregating data...")

    # 1. Create daily market summary
    daily_market_summary = pd.DataFrame()

    # Aggregate stock data by date
    if not stock_df.empty and 'date' in stock_df.columns:
        stock_daily = stock_df.groupby('date').agg({
            'daily_return': 'mean',
            'volatility_5d': 'mean',
            'Volume': 'sum'
        }).rename(columns={
            'daily_return': 'avg_stock_return',
            'volatility_5d': 'avg_stock_volatility',
            'Volume': 'total_stock_volume'
        })

        daily_market_summary = pd.concat([daily_market_summary, stock_daily], axis=1)

    # Aggregate crypto data by date
    if not crypto_df.empty and 'date' in crypto_df.columns:
        crypto_daily = crypto_df.groupby('date').agg({
            'current_price': 'mean',
            'market_cap': 'sum'
        }).rename(columns={
            'current_price': 'avg_crypto_price',
            'market_cap': 'total_crypto_market_cap'
        })

        daily_market_summary = pd.concat([daily_market_summary, crypto_daily], axis=1).fillna(0)

    # 2. Aggregate economic indicators
    if not economic_df.empty and 'date' in economic_df.columns:
        # Create pivot table for economic indicators
        economic_daily = economic_df.groupby('date').agg({
            'indicator_value': 'mean',
        }).rename(columns={
            'indicator_value': 'avg_indicator_value',
        })

        daily_market_summary = pd.concat([daily_market_summary, economic_daily], axis=1).ffill().fillna(0)

    # 3. Aggregate news sentiment
    if not news_df.empty and 'date' in news_df.columns:
        news_daily = news_df.groupby('date').agg({
            'sentiment_score': 'mean'
        }).rename(columns={
            'sentiment_score': 'avg_news_sentiment'
        })

        daily_market_summary = pd.concat([daily_market_summary, news_daily], axis=1).ffill().bfill()

    # 4. Join with quarterly financial reports (on nearest date)
    # This is more complex as reports are quarterly - we'll create a simplified version
    if not financial_reports_df.empty and 'date' in financial_reports_df.columns:
        # Get aggregate financials by date
        fin_quarterly = financial_reports_df.groupby('date').agg({
            'revenue': 'sum',
            'net_income': 'sum',
            'profit_margin': 'mean'
        }).rename(columns={
            'revenue': 'total_revenue',
            'net_income': 'total_net_income',
            'profit_margin': 'avg_profit_margin'
        })

        # Forward fill the quarterly data to have values for each day
        fin_quarterly = fin_quarterly.resample('D').ffill()

        # Join with daily summary
        daily_market_summary = pd.concat([daily_market_summary, fin_quarterly], axis=1)

    # Ensure index is datetime for proper joining
    daily_market_summary.index = pd.to_datetime(daily_market_summary.index)

    # Fill NaN values with appropriate methods
    daily_market_summary = daily_market_summary.ffill()

    print(f"Created aggregate market summary with {len(daily_market_summary)} days of data")
    return daily_market_summary


# 3. DATA LOADING FUNCTIONS

def load_to_database(daily_market_summary):
    """Load the transformed data into PostgreSQL database"""
    print("Loading data to PostgreSQL database...")

    # Connect to database
    conn = psycopg2.connect(**DB_CONFIG['postgres'])
    cursor = conn.cursor()

    # Create tables if they don't exist
    create_tables_sql = """
    -- Market summary table
    CREATE TABLE IF NOT EXISTS market_summary (
        date DATE PRIMARY KEY,
        avg_stock_return FLOAT,
        avg_stock_volatility FLOAT,
        total_stock_volume FLOAT,
        avg_crypto_price FLOAT,
        total_crypto_market_cap BIGINT,
        avg_news_sentiment FLOAT,
        avg_indicator_value FLOAT
    );
    """

    cursor.execute(create_tables_sql)
    conn.commit()

    # Define the columns that exist in the market_summary table
    valid_columns = [
        'avg_stock_return', 'avg_stock_volatility', 'total_stock_volume',
        'avg_crypto_price', 'total_crypto_market_cap', 'avg_news_sentiment',
        'avg_indicator_value'
    ]

    # Filter daily_market_summary to only include valid columns
    filtered_summary = daily_market_summary.filter(items=valid_columns)

    # Load market summary data
    for index, row in filtered_summary.iterrows():
        # Convert NaN to None for SQL compatibility
        row_dict = {k: None if pd.isna(v) else v for k, v in row.to_dict().items()}

        columns = ', '.join(row_dict.keys())
        placeholders = ', '.join(['%s'] * len(row_dict))

        insert_sql = f"""
        INSERT INTO market_summary (date, {columns})
        VALUES (%s, {placeholders})
        ON CONFLICT (date) DO UPDATE SET
            {', '.join([f"{k} = EXCLUDED.{k}" for k in row_dict.keys()])}
        """

        cursor.execute(insert_sql, [index.date()] + list(row_dict.values()))

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print("Data successfully loaded to database")


# 4. MAIN ETL PIPELINE FUNCTION

def run_etl_pipeline():
    """Execute the full ETL pipeline"""
    print("Starting ETL pipeline...")

    # EXTRACT
    try:
        # Extract data from all sources
        stock_df = extract_stock_data_csv('data/stock_market_data.csv')
        crypto_df = extract_crypto_data_api()
        economic_df = extract_economic_data_sql()
        news_df = extract_news_data_mongodb()
        financial_reports_df = extract_financial_reports_gdrive()
    except Exception as e:
        print(f"Error during extraction: {e}")
        return False

    # TRANSFORM
    try:
        # Clean individual datasets
        stock_df = clean_stock_data(stock_df)
        crypto_df = clean_crypto_data(crypto_df)
        economic_df = clean_economic_data(economic_df)
        news_df = clean_news_data(news_df)
        financial_reports_df = clean_financial_reports(financial_reports_df)

        # Standardize all datasets
        stock_df, crypto_df, economic_df, news_df, financial_reports_df = standardize_data(
            stock_df, crypto_df, economic_df, news_df, financial_reports_df)

        # Engineer features
        stock_df, crypto_df, economic_df, news_df, financial_reports_df = engineer_features(
            stock_df, crypto_df, economic_df, news_df, financial_reports_df)

        # Aggregate data
        daily_market_summary = aggregate_data(
            stock_df, crypto_df, economic_df, news_df, financial_reports_df)
    except Exception as e:
        print(f"Error during transformation: {e}")
        return False

    # LOAD
    try:
        load_to_database(daily_market_summary)
    except Exception as e:
        print(f"Error during loading: {e}")
        return False

    print("ETL pipeline completed successfully!")
    return True


# Run the pipeline
if __name__ == "__main__":
    run_etl_pipeline()
