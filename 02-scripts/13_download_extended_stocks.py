"""
DOWNLOAD EXTENDED STOCKS AND ETFS
==================================

Downloads additional 10 stocks + 4 new sector ETFs
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import time

# New stocks to download
NEW_STOCKS = ['AAPL', 'MSFT', 'GS', 'JNJ', 'WMT', 'PG', 'META', 'GOOGL', 'NEE', 'AMT']
NEW_ETFS = ['XLC', 'XLP', 'XLRE', 'XLU']

ALL_TICKERS = NEW_STOCKS + NEW_ETFS

START_DATE = '2019-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path('../01-data')


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download and process ticker data"""
    print(f"  Downloading {ticker}...")

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
    print("DOWNLOADING EXTENDED STOCKS AND ETFs")
    print("=" * 80)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {len(ALL_TICKERS)} ({len(NEW_STOCKS)} stocks + {len(NEW_ETFS)} ETFs)")
    print()

    results = {'success': [], 'failed': []}

    for i, ticker in enumerate(ALL_TICKERS, 1):
        print(f"[{i}/{len(ALL_TICKERS)}] {ticker}")

        df = download_ticker(ticker, START_DATE, END_DATE)

        if df is not None:
            results['success'].append(ticker)
        else:
            results['failed'].append(ticker)

        if i < len(ALL_TICKERS):
            time.sleep(1)
        print()

    # Summary
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(results['success'])}/{len(ALL_TICKERS)}")
    if results['success']:
        print(f"  {', '.join(results['success'])}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        print(f"  {', '.join(results['failed'])}")


if __name__ == "__main__":
    main()
