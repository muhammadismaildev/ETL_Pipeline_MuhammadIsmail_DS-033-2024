#!/usr/bin/env python3
import json
from pymongo import MongoClient

# ---------------------- CONFIG ----------------------
MONGO_URI = 'mongodb+srv://ismail:Windows12@cluster0.ox33b.mongodb.net/'
DB_NAME = 'mobility'
COLLECTION_NAME = 'locations'
JSON_FILE = 'locations.json'

# ---------------------- FUNCTION TO LOAD DATA ----------------------
def load_locations_to_db():
    try:
        # Establish connection to MongoDB
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        # Load JSON file data
        with open(JSON_FILE, 'r') as file:
            locations_data = json.load(file)
        
        if not isinstance(locations_data, list):
            raise ValueError("JSON file must contain a list of documents")
        
        # Clear existing data in the collection (optional)
        collection.delete_many({})
        print(f"Existing documents cleared from '{COLLECTION_NAME}' collection.")

        # Insert data into collection
        result = collection.insert_many(locations_data)
        print(f"Inserted {len(result.inserted_ids)} documents into '{COLLECTION_NAME}' collection.")

    except Exception as e:
        print("An error occurred while loading locations to MongoDB:", e)
    finally:
        client.close()

# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    load_locations_to_db()
