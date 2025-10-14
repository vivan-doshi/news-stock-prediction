"""
DOWNLOAD ALL 50 STOCKS + ETF DATA
===================================

Downloads stock price data for all 50 stocks across 10 sectors
using the configuration from 21_expanded_50_stock_config.py
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time
import sys

# Import stock configuration
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
config = import_module('21_expanded_50_stock_config')

EXPANDED_STOCKS = config.EXPANDED_STOCKS

# Date range - Updated to 2021-2025 based on EODHD news data availability analysis
START_DATE = '2021-01-01'
END_DATE = '2025-07-31'
DATA_DIR = Path('../01-data')
DATA_DIR.mkdir(exist_ok=True)

def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and process ticker data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"    ⚠ No data returned")
            return None

        df['Return'] = df['Close'].pct_change()
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].copy()
        df.index = df.index.tz_localize(None)

        print(f"    ✓ {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")

        # Save
        output_path = DATA_DIR / f"{ticker}_stock_data.csv"
        df.to_csv(output_path)
        print(f"    ✓ Saved to {output_path.name}")

        return df

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def main():
    print("=" * 80)
    print("DOWNLOADING ALL 50 STOCKS")
    print("=" * 80)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"Total stocks: {len(EXPANDED_STOCKS)}")
    print(f"Output directory: {DATA_DIR.absolute()}\n")

    # Get unique ETFs
    unique_etfs = set(info['etf'] for info in EXPANDED_STOCKS.values())
    print(f"Unique ETFs needed: {len(unique_etfs)}")
    print(f"  {', '.join(sorted(unique_etfs))}\n")

    # All tickers to download
    all_tickers = list(EXPANDED_STOCKS.keys()) + list(unique_etfs)

    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }

    # Group by sector for organized output
    sectors = {}
    for ticker, info in EXPANDED_STOCKS.items():
        sector = info['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append((ticker, info))

    # Download stocks by sector
    total = 0
    for sector in sorted(sectors.keys()):
        print(f"\n{'='*80}")
        print(f"{sector.upper()}")
        print(f"{'='*80}")

        for ticker, info in sorted(sectors[sector]):
            total += 1
            print(f"[{total}/{len(EXPANDED_STOCKS)}] {ticker} - {info['name']}")

            # Check if file exists
            output_path = DATA_DIR / f"{ticker}_stock_data.csv"
            if output_path.exists():
                print(f"    ⏭ Already exists, skipping")
                results['skipped'].append(ticker)
            else:
                df = download_ticker(ticker, START_DATE, END_DATE)

                if df is not None:
                    results['success'].append(ticker)
                else:
                    results['failed'].append(ticker)

                # Rate limiting
                time.sleep(0.5)

    # Download ETFs
    print(f"\n{'='*80}")
    print(f"SECTOR ETFs")
    print(f"{'='*80}")

    for i, etf in enumerate(sorted(unique_etfs), 1):
        print(f"[{i}/{len(unique_etfs)}] {etf}")

        output_path = DATA_DIR / f"{etf}_stock_data.csv"
        if output_path.exists():
            print(f"    ⏭ Already exists, skipping")
            results['skipped'].append(etf)
        else:
            df = download_ticker(etf, START_DATE, END_DATE)

            if df is not None:
                results['success'].append(etf)
            else:
                results['failed'].append(etf)

            if i < len(unique_etfs):
                time.sleep(0.5)

    # Summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"✓ Successful: {len(results['success'])}")
    print(f"⏭ Skipped (already exist): {len(results['skipped'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"Total processed: {len(results['success']) + len(results['skipped']) + len(results['failed'])}/{len(all_tickers)}")

    if results['failed']:
        print(f"\n❌ Failed tickers:")
        print(f"  {', '.join(results['failed'])}")

    print(f"\n📁 Data saved to: {DATA_DIR.absolute()}")

    return results

if __name__ == "__main__":
    results = main()
