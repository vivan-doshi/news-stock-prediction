"""
Master Script: Download All Data Using EODHD
Downloads news, stock prices, and market data for event study analysis
"""

import os
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import will be done dynamically to avoid issues
# from 00c_eodhd_news import EODHDNewsAcquisition

class DataDownloadPipeline:
    """Complete data download pipeline using EODHD"""

    def __init__(self, output_dir: str = "../01-data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load environment variables
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)

    def download_stock_data(self, ticker: str, start_date: str, end_date: str):
        """Download stock price data using yfinance"""
        print(f"\nDownloading {ticker} stock data...")
        print(f"  Date range: {start_date} to {end_date}")

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f"  ⚠ No stock data found for {ticker}")
                return None

            # Reset index to get date as column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'Date'})

            # Save
            filename = self.output_dir / f"{ticker}_stock_data.csv"
            df.to_csv(filename, index=False)

            print(f"  ✓ Downloaded {len(df)} trading days")
            print(f"  ✓ Saved to {filename}")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def download_market_data(self, start_date: str, end_date: str):
        """Download S&P 500 market index data"""
        print(f"\nDownloading S&P 500 index data...")
        print(f"  Date range: {start_date} to {end_date}")

        try:
            spy = yf.Ticker("^GSPC")  # S&P 500 index
            df = spy.history(start=start_date, end=end_date)

            if df.empty:
                print(f"  ⚠ No market data found")
                return None

            # Reset index and calculate returns
            df = df.reset_index()
            df['Market_Return'] = df['Close'].pct_change()

            # Save
            filename = self.output_dir / "sp500_market_data.csv"
            df.to_csv(filename, index=False)

            print(f"  ✓ Downloaded {len(df)} trading days")
            print(f"  ✓ Saved to {filename}")
            return df

        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def download_news_eodhd(self, ticker: str, start_date: str, end_date: str):
        """Download news data using EODHD"""
        print(f"\nDownloading {ticker} news from EODHD...")

        try:
            # Import here to avoid module naming issues
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "eodhd_news",
                Path(__file__).parent / "00c_eodhd_news.py"
            )
            eodhd_news = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eodhd_news)

            downloader = eodhd_news.EODHDNewsAcquisition(output_dir=str(self.output_dir))
            df = downloader.download_news(ticker, start_date, end_date, save=True)
            return df
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return None

    def run_full_pipeline(self,
                         tickers: list,
                         start_date: str,
                         end_date: str):
        """Run complete data download pipeline"""

        print("="*70)
        print("DATA DOWNLOAD PIPELINE - EODHD")
        print("="*70)
        print(f"\nTickers: {', '.join(tickers)}")
        print(f"Date Range: {start_date} to {end_date}")

        results = {}

        # Download market data (S&P 500)
        print("\n" + "="*70)
        print("[1/3] MARKET DATA (S&P 500)")
        print("="*70)
        results['market'] = self.download_market_data(start_date, end_date)

        # Download stock data for each ticker
        print("\n" + "="*70)
        print("[2/3] STOCK PRICE DATA")
        print("="*70)
        results['stocks'] = {}
        for ticker in tickers:
            results['stocks'][ticker] = self.download_stock_data(ticker, start_date, end_date)

        # Download news data for each ticker
        print("\n" + "="*70)
        print("[3/3] NEWS DATA (EODHD)")
        print("="*70)
        results['news'] = {}
        for ticker in tickers:
            results['news'][ticker] = self.download_news_eodhd(ticker, start_date, end_date)

        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD COMPLETE")
        print("="*70)
        print(f"\nData saved to: {self.output_dir.absolute()}")
        print("\nSummary:")

        if results['market'] is not None:
            print(f"  ✓ Market data: {len(results['market'])} days")

        for ticker in tickers:
            print(f"\n  {ticker}:")
            if results['stocks'].get(ticker) is not None:
                print(f"    ✓ Stock data: {len(results['stocks'][ticker])} days")
            if results['news'].get(ticker) is not None:
                print(f"    ✓ News articles: {len(results['news'][ticker])}")

        print("="*70)

        return results


if __name__ == "__main__":
    # Configuration
    TICKERS = ['AAPL', 'TSLA']

    # Date ranges matching your original data
    START_DATE = '2020-01-01'
    END_DATE_AAPL = '2024-01-31'  # AAPL original range
    END_DATE_TSLA = '2025-10-08'  # TSLA original range

    # Initialize pipeline
    pipeline = DataDownloadPipeline(output_dir='../01-data')

    print("\n" + "="*70)
    print("EODHD DATA DOWNLOAD - ORIGINAL DATE RANGES")
    print("="*70)
    print("\nNote: Downloads will use original date ranges:")
    print(f"  AAPL: {START_DATE} to {END_DATE_AAPL}")
    print(f"  TSLA: {START_DATE} to {END_DATE_TSLA}")
    print("\nEODHD Free Tier Limits:")
    print("  - 20 API calls per day")
    print("  - Each news request = 5 API calls")
    print("  - Max 4 news requests per day on free tier")
    print("\nFor full historical access, upgrade to paid plan ($19.99/month)")
    print("\nStarting download...")

    # Download AAPL data
    print("\n" + "="*70)
    print("DOWNLOADING APPLE (AAPL) DATA")
    print("="*70)
    results_aapl = pipeline.run_full_pipeline(
        tickers=['AAPL'],
        start_date=START_DATE,
        end_date=END_DATE_AAPL
    )

    # Download TSLA data
    print("\n" + "="*70)
    print("DOWNLOADING TESLA (TSLA) DATA")
    print("="*70)
    results_tsla = pipeline.run_full_pipeline(
        tickers=['TSLA'],
        start_date=START_DATE,
        end_date=END_DATE_TSLA
    )

    print("\n" + "="*70)
    print("ALL DOWNLOADS COMPLETE!")
    print("="*70)
    print(f"\nAll data saved to: {pipeline.output_dir.absolute()}")
