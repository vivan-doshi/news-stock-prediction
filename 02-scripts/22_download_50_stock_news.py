"""
Download news data for all 50 stocks in expanded configuration
"""

import os
import sys
import pandas as pd
from pathlib import Path
import time

# Import the expanded stock configuration
import importlib
config_module = importlib.import_module('21_expanded_50_stock_config')
EXPANDED_STOCKS = config_module.EXPANDED_STOCKS

# Import the news downloader
news_module = importlib.import_module('00c_eodhd_news')
download_news_for_ticker = news_module.download_news_for_ticker

DATA_DIR = Path("../01-data")

def check_existing_news():
    """Check which stocks already have news data"""
    existing = []
    missing = []

    for ticker in EXPANDED_STOCKS.keys():
        news_file = DATA_DIR / f"{ticker}_eodhd_news.csv"
        if news_file.exists():
            existing.append(ticker)
        else:
            missing.append(ticker)

    return existing, missing

def main():
    print("="*80)
    print("DOWNLOADING NEWS FOR 50-STOCK EXPANDED CONFIGURATION")
    print("="*80)

    existing, missing = check_existing_news()

    print(f"\n✅ Stocks with existing news data: {len(existing)}")
    print(f"❌ Stocks needing news data: {len(missing)}")

    if missing:
        print(f"\nMissing news for: {', '.join(missing)}\n")

        response = input(f"Download news for {len(missing)} stocks? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

        print("\nStarting downloads...")
        successful = []
        failed = []

        for i, ticker in enumerate(missing, 1):
            print(f"\n[{i}/{len(missing)}] Downloading {ticker}...")
            try:
                download_news_for_ticker(
                    ticker=ticker,
                    start_date='2019-01-01',
                    end_date='2024-12-31',
                    output_dir=str(DATA_DIR)
                )
                successful.append(ticker)
                print(f"  ✅ {ticker} complete")

                # Rate limiting
                if i < len(missing):
                    time.sleep(2)

            except Exception as e:
                print(f"  ❌ {ticker} failed: {str(e)}")
                failed.append(ticker)

        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        print(f"✅ Successful: {len(successful)}/{len(missing)}")
        if failed:
            print(f"❌ Failed: {len(failed)} - {', '.join(failed)}")
    else:
        print("\n✅ All stocks already have news data!")

    # Final check
    existing, missing = check_existing_news()
    print(f"\nFinal status: {len(existing)}/50 stocks have news data")

    if missing:
        print(f"Still missing: {', '.join(missing)}")

if __name__ == "__main__":
    main()
