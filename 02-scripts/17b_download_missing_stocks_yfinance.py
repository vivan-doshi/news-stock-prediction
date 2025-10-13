"""
Download missing stock price data for 37 stocks using yfinance (FREE)
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# Import the expanded config
sys.path.append(os.path.dirname(__file__))
from importlib import import_module
config = import_module('16_expanded_stock_config')

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
print("DOWNLOADING MISSING STOCK PRICE DATA (yfinance)")
print("=" * 80)
print(f"\nTotal stocks in config: {len(ALL_STOCKS)}")
print(f"Existing stocks: {len(EXISTING_STOCKS)}")
print(f"Missing stocks: {len(MISSING_STOCKS)}")
print(f"\nDate range: {START_DATE} to {END_DATE}")
print(f"API: Yahoo Finance (yfinance - FREE)")

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    try:
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            return None

        # Reset index to make Date a column
        df = df.reset_index()

        # Rename columns to match existing format
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Adjusted_close',  # Yahoo Finance Close is adjusted
            'Volume': 'Volume'
        })

        # Add Close column (same as Adjusted_close for consistency)
        df['Close'] = df['Adjusted_close']

        # Convert date to datetime (remove timezone if present)
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Select relevant columns in the right order
        columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted_close', 'Volume']
        df = df[columns]

        return df

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

print("\n" + "=" * 80)
print("DOWNLOADING STOCK DATA")
print("=" * 80)

successful = []
failed = []

for i, ticker in enumerate(MISSING_STOCKS, 1):
    sector = config.ALL_STOCKS[ticker]['sector']
    print(f"\n[{i}/{len(MISSING_STOCKS)}] {ticker} ({sector})")

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

    # Be nice to Yahoo - small delay
    time.sleep(0.2)

print("\n" + "=" * 80)
print("DOWNLOAD SUMMARY")
print("=" * 80)
print(f"\n✓ Successful: {len(successful)}/{len(MISSING_STOCKS)}")
print(f"✗ Failed: {len(failed)}/{len(MISSING_STOCKS)}")

if successful:
    print(f"\nSuccessful downloads:")
    for ticker in successful:
        sector = config.ALL_STOCKS[ticker]['sector']
        print(f"  • {ticker:6} ({sector})")

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
    print("\n✓ Ready to proceed with news download and experiments!")
else:
    print(f"\n⚠️ Still missing {total_expected - total_now} stocks")

print("\n" + "=" * 80)
