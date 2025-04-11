# Integrated Financial Data ETL Pipeline (ETL_Pipeline_MuhammadIsmail_DS-033-2024)

## Overview

This project implements an Extract, Transform, Load (ETL) pipeline designed to gather financial data from various sources, process it, and load it into a unified PostgreSQL database. The goal is to create a consolidated dataset (`market_summary`) suitable for financial analysis and market insights.

The pipeline extracts data from:
* CSV files (Stock Market Data)
* Public APIs (Cryptocurrency Data - CoinGecko)
* SQL Databases (Economic Indicators - PostgreSQL)
* NoSQL Databases (Financial News Sentiment - MongoDB)
* Google Sheets (Company Financial Reports)

It performs cleaning, standardization, feature engineering, and aggregation before loading the final results. The pipeline includes scheduling capabilities and integrates with CI/CD workflows using GitHub Actions for automated testing and validation.

## Features

* **Multi-Source Extraction:** Connects to diverse data sources including files, APIs, SQL, NoSQL, and Google Sheets.
* **Data Cleaning & Standardization:** Handles missing values, duplicates, data type conversions, date standardization, and currency conversion attempts.
* **Feature Engineering:** Creates derived metrics like daily returns, moving averages, volatility, sentiment averages, and financial ratios.
* **Data Aggregation:** Consolidates data from all sources into a daily market summary.
* **Automated Loading:** Loads processed data into a PostgreSQL database using an upsert strategy.
* **Scheduling:** Includes a script (`scheduler.py`) to run the pipeline automatically on a schedule (e.g., daily).
* **Configuration Management:** Separates sensitive information (API keys, DB credentials) into configuration files.
* **CI/CD Integration:** Includes a GitHub Actions workflow (`.github/workflows/ci_cd.yml`) for automated linting and testing (requires test implementation).

## Technology Stack

* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **API Interaction:** Requests
* **Database Interaction:**
    * PostgreSQL: psycopg2, SQLAlchemy
    * MongoDB: pymongo
* **Google Sheets:** gspread, oauth2client
* **Scheduling:** schedule
* **Databases:** PostgreSQL (Source & Target), MongoDB (Source)
* **Configuration:** JSON
* **CI/CD:** GitHub Actions

## Setup and Installation

**1. Prerequisites:**
* Python 3.8+
* Git
* Access to a PostgreSQL server (for source economic data and target `market_summary` table)
* Access to a MongoDB server (for source financial news data)
* CoinGecko API Key (Free tier available)
* Google Cloud Project with Sheets API and Drive API enabled, and a Service Account key file (`gspread-creds.json`).

**2. Clone the Repository:**
```bash
git clone <your-repository-url>
cd ETL_Pipeline_<YourName>_<RollNumber>
3. Create a Virtual Environment (Recommended):python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
4. Install Dependencies:pip install -r requirements.txt
5. Configure Environment:Database Configuration (config/db_config.json):Create this file and populate it with your actual database connection details. DO NOT COMMIT REAL CREDENTIALS.{
  "postgres": {
    "host": "your_postgres_host",
    "port": 5432,
    "user": "your_postgres_user",
    "password": "your_postgres_password",
    "database": "your_database_name"
  },
  "mongodb": {
    "host": "your_mongodb_host",
    "port": 27017,
    "db": "your_mongodb_database"
    // Add user/password if authentication is enabled
    // "user": "your_mongo_user",
    // "password": "your_mongo_password"
  }
}
API Keys (config/api_keys.json):Create this file with your CoinGecko API key. DO NOT COMMIT REAL KEYS.{
  "coingecko": "YOUR_COINGECKO_API_KEY"
}
Google Sheets Credentials (config/gspread-creds.json):Download the JSON key file for your Google Service Account (with Sheets and Drive API permissions) and place it in the config/ directory named gspread-creds.json. Ensure this service account has access to the target Google Sheet. DO NOT COMMIT THIS FILE. Add config/gspread-creds.json to your .gitignore file.6. Database Setup:Ensure your source PostgreSQL and MongoDB instances are running.Populate the source PostgreSQL economic_indicators table and the MongoDB financial_news collection with sample data if necessary.Ensure the target PostgreSQL database specified in db_config.json exists. The etl_pipeline.py script will attempt to create the market_summary table if it doesn't exist.7. Data Files:Place the required input CSV file (stock_market_data.csv) in the data/ directory. Ensure it has the columns: Date, Open, High, Low, Close, Volume, Symbol.Ensure the Google Sheet specified in the code (financial_reports) exists and is accessible by your service account.Usage1. Manual Pipeline Run:To execute the ETL pipeline once:python etl_pipeline.py
Check the console output for progress and any errors. The processed data will be loaded into the target PostgreSQL market_summary table.2. Running the Scheduler:To start the scheduler which runs the pipeline automatically based on the defined schedule (daily at 1:00 AM by default):python scheduler.py
This script will run continuously in the foreground, checking every minute if a scheduled job is due. Keep the terminal open or run it as a background process (e.g., using nohup or a process manager like systemd or supervisor) for persistent scheduling.Pipeline DetailsThe etl_pipeline.py script orchestrates the process:Extract: Functions like extract_stock_data_csv, extract_crypto_data_api, etc., fetch data.Transform: Functions like clean_stock_data, standardize_data, engineer_features, aggregate_data process the data.Load: The load_to_database function writes the final daily_market_summary DataFrame to the PostgreSQL target table.CI/CDThis project uses GitHub Actions for Continuous Integration. The workflow definition can be found in .github/workflows/ci_cd.yml. Upon pushes or pull requests, it typically performs:Code CheckoutDependency InstallationLinting (e.g., using Flake8)Testing (Requires implementation of tests, e.g., using pytest)This helps ensure code quality and prevent regressions.ContributingContributions are welcome! Please follow standard Gitflow practices:Fork the repository.Create a new feature branch (git checkout -b feature/your-feature-name).Make your changes.Ensure code is linted and tests pass (if applicable).Commit your changes (git commit -m 'Add some feature').Push to the branch (git push origin feature/your-feature-name).Open