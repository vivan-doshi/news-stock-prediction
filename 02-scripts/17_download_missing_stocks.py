"""
Download missing stock price data for 37 stocks
Using EODHD API (same as existing data)
"""

import os
import sys
import pandas as pd
import requests
from datetime import datetime
import time

# Import the expanded config
sys.path.append(os.path.dirname(__file__))
from importlib import import_module
config = import_module('16_expanded_stock_config')

# EODHD API configuration
API_KEY = '67604bf0e70f08.47044417'
BASE_URL = 'https://eodhd.com/api/eod'

# Data parameters
START_DATE = '2022-01-01'
END_DATE = '2024-12-31'
DATA_DIR = '01-data'

# List of stocks we already have
EXISTING_STOCKS = [
    'AAPL', 'AMZN', 'AMT', 'BA', 'GOOGL', 'GS', 'JNJ', 'JPM',
    'META', 'MSFT', 'NEE', 'NVDA', 'PFE', 'PG', 'TSLA', 'WMT', 'XOM'
]

# Get missing stocks
ALL_STOCKS = list(config.ALL_STOCKS.keys())
MISSING_STOCKS = [s for s in ALL_STOCKS if s not in EXISTING_STOCKS]

print("=" * 80)
print("DOWNLOADING MISSING STOCK PRICE DATA")
print("=" * 80)
print(f"\nTotal stocks in config: {len(ALL_STOCKS)}")
print(f"Existing stocks: {len(EXISTING_STOCKS)}")
print(f"Missing stocks: {len(MISSING_STOCKS)}")
print(f"\nDate range: {START_DATE} to {END_DATE}")
print(f"API: EODHD")

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from EODHD API"""
    url = f"{BASE_URL}/{ticker}.US"
    params = {
        'api_token': API_KEY,
        'from': start_date,
        'to': end_date,
        'period': 'd',
        'fmt': 'json'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if not data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Rename columns to match existing format
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjusted_close': 'Adjusted_close',
            'volume': 'Volume'
        })

        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Select relevant columns
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted_close', 'Volume']
        df = df[columns]

        return df

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading {ticker}: {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error processing {ticker}: {e}")
        return None

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("\n" + "=" * 80)
print("DOWNLOADING STOCK DATA")
print("=" * 80)

successful = []
failed = []

for i, ticker in enumerate(MISSING_STOCKS, 1):
    print(f"\n[{i}/{len(MISSING_STOCKS)}] {ticker} ({config.ALL_STOCKS[ticker]['sector']})")

    # Download data
    df = download_stock_data(ticker, START_DATE, END_DATE)

    if df is not None and len(df) > 0:
        # Save to CSV
        output_file = os.path.join(DATA_DIR, f'{ticker}_stock_data.csv')
        df.to_csv(output_file, index=False)

        print(f"  ✓ Downloaded {len(df)} records")
        print(f"  ✓ Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
        print(f"  ✓ Saved to: {output_file}")

        successful.append(ticker)
    else:
        print(f"  ✗ Failed to download data")
        failed.append(ticker)

    # Rate limiting - wait 0.5 seconds between requests
    if i < len(MISSING_STOCKS):
        time.sleep(0.5)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"\n✓ Successful: {len(successful)}/{len(MISSING_STOCKS)}")
print(f"✗ Failed: {len(failed)}/{len(MISSING_STOCKS)}")

if successful:
    print(f"\nSuccessful downloads:")
    for ticker in successful:
        print(f"  • {ticker}")

if failed:
    print(f"\n⚠️ Failed downloads:")
    for ticker in failed:
        print(f"  • {ticker}")

print("\n" + "=" * 80)
print("OVERALL DATA STATUS")
print("=" * 80)

total_expected = len(ALL_STOCKS)
total_existing = len(EXISTING_STOCKS)
total_downloaded = len(successful)
total_now = total_existing + total_downloaded

print(f"\nTotal stocks expected: {total_expected}")
print(f"Previously existing: {total_existing}")
print(f"Just downloaded: {total_downloaded}")
print(f"Total now: {total_now}")
print(f"Completion: {total_now}/{total_expected} ({100*total_now/total_expected:.1f}%)")

if total_now == total_expected:
    print("\n🎉 All stock data downloaded successfully!")
else:
    print(f"\n⚠️ Still missing {total_expected - total_now} stocks")

print("\n" + "=" * 80)
