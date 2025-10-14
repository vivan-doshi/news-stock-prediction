"""
DOWNLOAD 2025 NEWS DATA UPDATE
================================

Downloads news data for all 50 stocks from January 1, 2025 to July 1, 2025
and merges with existing data.

This script:
1. Downloads new 2025 news data (Jan 1 - Jul 1, 2025)
2. Merges with existing data (2019-2024)
3. Applies 3:30 PM ET cutoff rule (shift to next trading day if after 3:30 PM)
4. Saves updated files

Author: Data Update Pipeline
Date: 2025-10-13
"""

import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Import configuration
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

config = import_module('21_expanded_50_stock_config')
news_module = import_module('00c_eodhd_news')

EXPANDED_STOCKS = config.EXPANDED_STOCKS
EODHDNewsAcquisition = news_module.EODHDNewsAcquisition

DATA_DIR = Path("../01-data")
BACKUP_DIR = Path("../backup_old_data")

# New date range for 2025 data
START_DATE_2025 = '2025-01-01'
END_DATE_2025 = '2025-07-01'

def create_backup():
    """Create backup of existing data files"""
    print("\n" + "="*80)
    print("CREATING BACKUP OF EXISTING DATA")
    print("="*80)

    BACKUP_DIR.mkdir(exist_ok=True, parents=True)

    backup_count = 0
    for ticker in EXPANDED_STOCKS.keys():
        old_file = DATA_DIR / f"{ticker}_eodhd_news.csv"
        if old_file.exists():
            backup_file = BACKUP_DIR / f"{ticker}_eodhd_news_backup.csv"

            # Only backup if not already backed up
            if not backup_file.exists():
                import shutil
                shutil.copy2(old_file, backup_file)
                backup_count += 1

    print(f"✅ Backed up {backup_count} files to {BACKUP_DIR}")

def adjust_news_time_cutoff(df, ticker):
    """
    Adjust news dates based on 3:30 PM ET cutoff rule.

    Rule: News published after 3:30 PM ET should be counted as next trading day's news
    News cycle: 3:30 PM ET to 3:30 PM ET (not midnight to midnight)

    Parameters:
    - df: DataFrame with 'date' and 'time' columns
    - ticker: Stock ticker for logging

    Returns:
    - DataFrame with adjusted 'event_date' column
    """
    print(f"  → Applying 3:30 PM ET cutoff rule for {ticker}...")

    # Combine date and time into datetime
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')

    # Convert to ET timezone (handle both EST and EDT)
    eastern = pytz.timezone('US/Eastern')

    # If datetime doesn't have timezone, assume UTC and convert
    def convert_to_et(dt):
        if pd.isna(dt):
            return dt
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        return dt.astimezone(eastern)

    df['datetime_et'] = df['datetime'].apply(convert_to_et)

    # Extract hour in ET
    df['hour_et'] = df['datetime_et'].dt.hour
    df['minute_et'] = df['datetime_et'].dt.minute

    # Create event_date column
    df['event_date'] = df['date']

    # Shift to next day if after 3:30 PM ET (15:30)
    after_cutoff = (df['hour_et'] > 15) | ((df['hour_et'] == 15) & (df['minute_et'] >= 30))

    # For news after cutoff, shift to next calendar day
    df.loc[after_cutoff, 'event_date'] = df.loc[after_cutoff, 'date'] + pd.Timedelta(days=1)

    shifted_count = after_cutoff.sum()
    print(f"    • Shifted {shifted_count} articles ({shifted_count/len(df)*100:.1f}%) to next day")

    return df

def download_2025_news(ticker, downloader):
    """Download 2025 news for a specific ticker"""
    try:
        df_2025 = downloader.download_news(
            ticker=ticker,
            start_date=START_DATE_2025,
            end_date=END_DATE_2025,
            save=False  # Don't save yet, we'll merge first
        )

        if df_2025 is not None and len(df_2025) > 0:
            return df_2025
        else:
            return None

    except Exception as e:
        print(f"  ❌ Error downloading 2025 data: {str(e)}")
        return None

def merge_and_update(ticker):
    """Merge 2025 data with existing data and update file"""
    old_file = DATA_DIR / f"{ticker}_eodhd_news.csv"

    print(f"\n{'='*60}")
    print(f"{ticker} - {EXPANDED_STOCKS[ticker]['name']}")
    print(f"{'='*60}")

    # Load existing data
    if not old_file.exists():
        print(f"  ⚠️  No existing data file found")
        return False

    df_old = pd.read_csv(old_file)
    df_old['date'] = pd.to_datetime(df_old['date'])
    print(f"  ✓ Loaded existing data: {len(df_old):,} articles")
    print(f"    Date range: {df_old['date'].min()} to {df_old['date'].max()}")

    # Download 2025 data
    print(f"  → Downloading 2025 news data...")
    downloader = EODHDNewsAcquisition(output_dir=str(DATA_DIR))
    df_2025 = download_2025_news(ticker, downloader)

    if df_2025 is None or len(df_2025) == 0:
        print(f"  ⚠️  No 2025 data available")
        return False

    print(f"  ✓ Downloaded 2025 data: {len(df_2025):,} articles")

    # Merge data
    df_combined = pd.concat([df_old, df_2025], ignore_index=True)

    # Remove duplicates (by date + title)
    df_combined = df_combined.drop_duplicates(subset=['date', 'title'], keep='first')

    # Sort by date
    df_combined = df_combined.sort_values('date').reset_index(drop=True)

    print(f"  ✓ Merged data: {len(df_combined):,} total articles (removed {len(df_old)+len(df_2025)-len(df_combined)} duplicates)")
    print(f"    Final date range: {df_combined['date'].min()} to {df_combined['date'].max()}")

    # Apply 3:30 PM ET cutoff rule
    df_combined = adjust_news_time_cutoff(df_combined, ticker)

    # Save updated file
    df_combined.to_csv(old_file, index=False)
    print(f"  ✅ Saved updated file: {old_file}")

    return True

def main():
    print("="*80)
    print("DOWNLOADING 2025 NEWS DATA UPDATE")
    print("="*80)
    print(f"Date range: {START_DATE_2025} to {END_DATE_2025}")
    print(f"Stocks: {len(EXPANDED_STOCKS)}")

    # Create backup first
    create_backup()

    # Process each stock
    successful = []
    failed = []
    no_new_data = []

    for i, ticker in enumerate(sorted(EXPANDED_STOCKS.keys()), 1):
        print(f"\n[{i}/{len(EXPANDED_STOCKS)}]", end=" ")

        result = merge_and_update(ticker)

        if result:
            successful.append(ticker)
        elif result is False:
            no_new_data.append(ticker)
        else:
            failed.append(ticker)

        # Rate limiting
        if i < len(EXPANDED_STOCKS):
            time.sleep(3)  # 3 seconds between requests

    # Summary
    print("\n" + "="*80)
    print("UPDATE SUMMARY")
    print("="*80)
    print(f"✅ Successfully updated: {len(successful)}/{len(EXPANDED_STOCKS)}")
    print(f"⚠️  No new data: {len(no_new_data)}")
    print(f"❌ Failed: {len(failed)}")

    if successful:
        print(f"\n✅ Successfully updated ({len(successful)}):")
        for ticker in successful:
            print(f"  {ticker}")

    if no_new_data:
        print(f"\n⚠️  No new data ({len(no_new_data)}):")
        for ticker in no_new_data:
            print(f"  {ticker}")

    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for ticker in failed:
            print(f"  {ticker}")

    print(f"\n💾 Backup location: {BACKUP_DIR}")

if __name__ == "__main__":
    main()
