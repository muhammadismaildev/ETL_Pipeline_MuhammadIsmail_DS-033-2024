import pandas as pd
import requests
import datetime
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import time
from pymongo import MongoClient

# ---------------------- CONFIG ----------------------
CSV_FILE = 'green_tripdata_2023-01.csv'
WEATHER_API_KEY = 'd81a114871a599d9c7dc0f70954c314f'
WEATHER_URL = 'http://api.openweathermap.org/data/2.5/weather'
REALTIME_CONGESTION_API = 'http://localhost:5000/api/congestion?zone='
GOOGLE_SHEET_NAME = 'Zone Metadata'
MONGO_URI = 'mongodb+srv://ismail:Windows12@cluster0.ox33b.mongodb.net/'
DB_NAME = 'mobility'
COLLECTION_NAME = 'cleaned_trip_data'

# ---------------------- EXTRACT ----------------------
def load_csv_data():
    df = pd.read_csv(CSV_FILE)
    return df

def get_weather(lat, lon):
    params = {'lat': lat, 'lon': lon, 'appid': WEATHER_API_KEY, 'units': 'metric'}
    response = requests.get(WEATHER_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed']
        }
    return {'temp': None, 'humidity': None, 'wind_speed': None}

def load_google_sheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('gspread-creds.json', scope)
    client = gspread.authorize(creds)
    sheet = client.open(GOOGLE_SHEET_NAME).sheet1
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def get_congestion_level(zone_id):
    response = requests.get(f'{REALTIME_CONGESTION_API}{zone_id}')
    if response.status_code == 200:
        return response.json().get('congestion_level', None)
    return None

def load_static_metadata():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db['locations']
    data = list(collection.find())
    return pd.DataFrame(data)

# ---------------------- TRANSFORM ----------------------
def clean_data(df):
    df = df.dropna(subset=['fare_amount', 'tip_amount'])
    df = df[df['fare_amount'] > 0]
    df = df[df['tip_amount'] > 0]
    return df

def convert_timestamps(df):
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime']).dt.tz_localize('US/Eastern').dt.tz_convert('UTC')
    df['lpep_pickup_datetime'] = df['lpep_pickup_datetime'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return df

def enrich_data(df, weather_data, zone_meta, location_meta):
    df = df.merge(zone_meta, how='left', left_on='PULocationID', right_on='zone_id')
    df = df.merge(location_meta, how='left', on='zone_id')
    df['temperature'] = df['PULocationID'].apply(lambda z: weather_data.get(z, {}).get('temp'))
    df['humidity'] = df['PULocationID'].apply(lambda z: weather_data.get(z, {}).get('humidity'))
    df['wind_speed'] = df['PULocationID'].apply(lambda z: weather_data.get(z, {}).get('wind_speed'))
    df['congestion_level'] = df['PULocationID'].apply(get_congestion_level)
    df['weather_impact_score'] = (df['temperature'] * df['humidity']) / (df['wind_speed'] + 1)
    return df

# ---------------------- LOAD ----------------------
def load_to_mongodb(df):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    records = df.to_dict(orient='records')
    collection.delete_many({})
    collection.insert_many(records)

# ---------------------- MAIN ETL ----------------------
def run_etl():
    print("Starting ETL...")
    raw_df = load_csv_data()
    raw_df = raw_df.rename(columns={'tpep_pickup_datetime': 'pickup_datetime', 'PULocationID': 'PULocationID'})
    
    print("Cleaning data...")
    cleaned_df = clean_data(raw_df)

    print("Converting timestamps...")
    cleaned_df = convert_timestamps(cleaned_df)

    print("Loading Google Sheet metadata...")
    zone_meta = load_google_sheet()

    print("Loading MongoDB static metadata...")
    location_meta = load_static_metadata()
 
    print("Fetching weather data...")
    weather_data = {}
    for zone_id in cleaned_df['PULocationID'].unique()[0:2]:
        weather_data[zone_id] = get_weather(lat=40.7128, lon=-74.0060)
        time.sleep(1)  # Avoid hitting API rate limit

    print("Enriching data...")
    enriched_df = enrich_data(cleaned_df, weather_data, zone_meta, location_meta)

    print("Loading to MongoDB...")
    load_to_mongodb(enriched_df)

    print("ETL Complete.")

if __name__ == '__main__':
    run_etl()
