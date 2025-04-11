# PostgreSQL Data Loader for Economic Indicators
# This script loads economic indicators data into a PostgreSQL database

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
import csv
import json


def create_economic_indicators_table(conn):
    """Create the economic_indicators table if it doesn't exist"""
    cursor = conn.cursor()

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS economic_indicators (
        id SERIAL PRIMARY KEY,
        indicator_date DATE NOT NULL,
        indicator_name VARCHAR(100) NOT NULL,
        indicator_value NUMERIC(12, 4) NOT NULL,
        region VARCHAR(50) NOT NULL,
        frequency VARCHAR(20) NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- Create indexes for query optimization
    CREATE INDEX IF NOT EXISTS idx_econ_date_name ON economic_indicators(indicator_date, indicator_name);
    CREATE INDEX IF NOT EXISTS idx_indicator_region ON economic_indicators(indicator_name, region);
    """

    cursor.execute(create_table_sql)
    conn.commit()
    cursor.close()
    print("Economic indicators table created or verified.")


def load_economic_data_manually(conn, data_list):
    """Load economic indicators data from a Python list into PostgreSQL"""
    cursor = conn.cursor()

    try:
        # Use efficient bulk insert
        insert_query = """
        INSERT INTO economic_indicators 
        (indicator_date, indicator_name, indicator_value, region, frequency)
        VALUES (%s, %s, %s, %s, %s)
        """

        cursor.executemany(insert_query, data_list)
        conn.commit()

        print(f"Successfully loaded {len(data_list)} economic indicator records manually")

    except Exception as e:
        conn.rollback()
        print(f"Error loading data manually: {e}")
    finally:
        cursor.close()


def generate_sample_economic_data():
    """Generate sample economic indicator data"""
    # Sample data structure: (date, indicator_name, value, region, frequency)
    sample_data = []

    # Current date for reference
    base_date = datetime(2025, 4, 4)

    # Generate Interest Rate data (monthly)
    regions = ['US', 'EU', 'Japan', 'UK']
    interest_rates = {'US': 3.75, 'EU': 2.50, 'Japan': 1.25, 'UK': 4.25}

    for i in range(12):  # 3 months
        month_date = base_date - timedelta(days= i)
        for region in regions:
            # Small random variation for previous months
            rate = interest_rates[region] + (0.25 * i * (1 if np.random.random() > 0.5 else -1))
            sample_data.append((
                month_date.strftime('%Y-%m-%d'),
                'Interest_Rate',
                round(rate, 2),
                region,
                'Monthly'
            ))

    # Generate Inflation Rate data (monthly)
    inflation_rates = {'US': 2.85, 'EU': 2.30, 'Japan': 1.20, 'UK': 3.10}

    for i in range(3):  # 3 months
        month_date = base_date - timedelta(days=30 * i)
        for region in regions:
            # Small random variation for previous months
            rate = inflation_rates[region] + (0.15 * i * (1 if np.random.random() > 0.5 else -1))
            sample_data.append((
                month_date.strftime('%Y-%m-%d'),
                'Inflation_Rate',
                round(rate, 2),
                region,
                'Monthly'
            ))

    # Generate Unemployment Rate data (monthly)
    unemployment_rates = {'US': 3.60, 'EU': 5.75, 'Japan': 2.35, 'UK': 4.10}

    for i in range(3):  # 3 months
        month_date = base_date - timedelta(days=30 * i)
        for region in regions:
            # Small random variation for previous months
            rate = unemployment_rates[region] + (0.15 * i * (1 if np.random.random() > 0.5 else -1))
            sample_data.append((
                month_date.strftime('%Y-%m-%d'),
                'Unemployment_Rate',
                round(rate, 2),
                region,
                'Monthly'
            ))

    # Generate GDP Growth data (quarterly)
    gdp_growth = {'US': 2.35, 'EU': 1.45, 'Japan': 0.95, 'UK': 1.85}

    # Current quarter
    quarter_date = datetime(2025, 3, 31)
    for region in regions:
        sample_data.append((
            quarter_date.strftime('%Y-%m-%d'),
            'GDP_Growth',
            gdp_growth[region],
            region,
            'Quarterly'
        ))

    # Previous quarter
    prev_quarter = datetime(2024, 12, 31)
    for region in regions:
        # Slightly lower growth in previous quarter
        rate = gdp_growth[region] - 0.15
        sample_data.append((
            prev_quarter.strftime('%Y-%m-%d'),
            'GDP_Growth',
            round(rate, 2),
            region,
            'Quarterly'
        ))

    # Generate Manufacturing PMI data (monthly)
    pmi_values = {'US': 53.20, 'EU': 49.80, 'Japan': 51.50, 'UK': 50.70}

    for i in range(3):  # 3 months
        month_date = base_date - timedelta(days=30 * i)
        for region in regions:
            # Small random variation for previous months
            pmi = pmi_values[region] - (0.4 * i * (1 if np.random.random() > 0.5 else -1))
            sample_data.append((
                month_date.strftime('%Y-%m-%d'),
                'Manufacturing_PMI',
                round(pmi, 2),
                region,
                'Monthly'
            ))

    return sample_data


def generate_csv_from_data(data_list, output_file="economic_indicators.csv"):
    """Generate a CSV file from the sample data"""
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header
        csv_writer.writerow(['indicator_date', 'indicator_name', 'indicator_value', 'region', 'frequency'])

        # Write data rows
        csv_writer.writerows(data_list)

    print(f"Generated CSV file: {output_file}")
    return output_file


def main():
    """Main function to demonstrate loading economic data into PostgreSQL"""

    with open('config/db_config.json') as f:
        DB_CONFIG = json.load(f)

    # Database connection parameters
    db_params = DB_CONFIG['postgres']

    try:
        # Connect to PostgreSQL
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_params)

        # Create table if it doesn't exist
        create_economic_indicators_table(conn)

        data = generate_sample_economic_data()
        load_economic_data_manually(conn, data)

        # Verify data was loaded
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM economic_indicators")
        count = cursor.fetchone()[0]
        print(f"Total records in economic_indicators table: {count}")
        cursor.close()

    except Exception as e:
        print(f"Database connection error: {e}")
    finally:
        # Close connection
        if 'conn' in locals() and conn is not None:
            conn.close()
            print("Database connection closed")


if __name__ == "__main__":
    main()