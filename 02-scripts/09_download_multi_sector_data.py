"""
MULTI-SECTOR STOCK AND ETF DATA ACQUISITION
============================================

Downloads data for 6 major stocks across different sectors plus their corresponding sector ETFs.
This will enable us to study news-driven deviations from sector performance.

Stocks & Sector ETFs:
- NVDA (Tech) → XLK (Technology Select Sector SPDR)
- JPM (Finance) → XLF (Financial Select Sector SPDR)
- PFE (Healthcare) → XLV (Health Care Select Sector SPDR)
- XOM (Energy) → XLE (Energy Select Sector SPDR)
- AMZN (Consumer Discretionary) → XLY (Consumer Discretionary SPDR)
- BA (Industrials) → XLI (Industrial Select Sector SPDR)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Configuration
STOCK_SECTOR_MAP = {
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'}
}

# Date range (5 years of data)
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

DATA_DIR = Path('../01-data')
DATA_DIR.mkdir(exist_ok=True)


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and process stock data"""
    print(f"  Downloading {ticker}...")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"    ⚠ No data returned for {ticker}")
            return None

        # Calculate returns
        df['Return'] = df['Close'].pct_change()

        # Keep only essential columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].copy()

        # Remove timezone info
        df.index = df.index.tz_localize(None)

        print(f"    ✓ Downloaded {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
        print(f"    ✓ Valid returns: {df['Return'].notna().sum()}")

        return df

    except Exception as e:
        print(f"    ❌ Error downloading {ticker}: {e}")
        return None


def save_stock_data(df: pd.DataFrame, ticker: str):
    """Save stock data to CSV"""
    if df is not None and not df.empty:
        output_path = DATA_DIR / f"{ticker}_stock_data.csv"
        df.to_csv(output_path)
        print(f"    ✓ Saved to {output_path}")
        return True
    return False


def main():
    """Download all stocks and sector ETFs"""
    print("=" * 80)
    print("MULTI-SECTOR DATA ACQUISITION")
    print("=" * 80)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"Output Directory: {DATA_DIR.absolute()}\n")

    all_tickers = []

    # Collect all tickers (stocks + ETFs)
    for stock, info in STOCK_SECTOR_MAP.items():
        all_tickers.append(stock)
        all_tickers.append(info['etf'])

    print(f"Tickers to download: {', '.join(all_tickers)}\n")

    results = {'success': [], 'failed': []}

    # Download each ticker
    for i, ticker in enumerate(all_tickers, 1):
        print(f"[{i}/{len(all_tickers)}] {ticker}")

        df = download_stock_data(ticker, START_DATE, END_DATE)

        if save_stock_data(df, ticker):
            results['success'].append(ticker)
        else:
            results['failed'].append(ticker)

        # Rate limiting
        if i < len(all_tickers):
            time.sleep(1)

        print()

    # Summary
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(results['success'])}/{len(all_tickers)}")
    if results['success']:
        print(f"  {', '.join(results['success'])}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        print(f"  {', '.join(results['failed'])}")

    # Create summary table
    print("\n" + "=" * 80)
    print("STOCK-SECTOR MAPPING")
    print("=" * 80)
    summary_data = []
    for stock, info in STOCK_SECTOR_MAP.items():
        summary_data.append({
            'Stock': stock,
            'Sector': info['sector'],
            'Sector_ETF': info['etf'],
            'Stock_Downloaded': 'Yes' if stock in results['success'] else 'No',
            'ETF_Downloaded': 'Yes' if info['etf'] in results['success'] else 'No'
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save mapping
    summary_df.to_csv(DATA_DIR / 'stock_sector_mapping.csv', index=False)
    print(f"\n✓ Saved mapping to {DATA_DIR / 'stock_sector_mapping.csv'}")


if __name__ == "__main__":
    main()
