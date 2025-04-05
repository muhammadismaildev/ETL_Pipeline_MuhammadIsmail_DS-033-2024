import schedule
import time
from etl_pipeline import run_etl

# Schedule it to run daily at 1:00 AM
schedule.every().day.at("01:00").do(run_etl)

print("Scheduler started. Waiting for next run...")

while True:
    schedule.run_pending()
    time.sleep(60)  # check every 60 seconds