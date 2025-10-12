"""
Finnhub Historical News Download Script
Downloads news data in yearly batches to work with free tier limits
"""

import os
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import time

class FinnhubHistoricalDownloader:
    """Downloads historical news from Finnhub in batches"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "../01-data"):
        """Initialize Finnhub downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load API key
        if api_key is None:
            env_path = Path(__file__).parent.parent / '.env'
            load_dotenv(env_path)
            api_key = os.getenv('FINNHUB_API_KEY')

        if not api_key:
            raise ValueError("FINNHUB_API_KEY not found. Add it to .env file.")

        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def download_news_batch(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download news for a specific date range

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'

        Returns:
        --------
        pd.DataFrame with news articles
        """
        print(f"  {start_date} to {end_date}...", end=" ")

        url = f"{self.base_url}/company-news"
        params = {
            'symbol': symbol,
            'from': start_date,
            'to': end_date,
            'token': self.api_key
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            news_data = response.json()

            if not news_data:
                print("No data")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(news_data)

            # Convert Unix timestamp to datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df['date'] = df['datetime'].dt.date
            df['date'] = pd.to_datetime(df['date'])

            # Add ticker column
            df['ticker'] = symbol

            # Sort by date
            df.sort_values('datetime', inplace=True)

            print(f"{len(df)} articles")

            # Small delay to respect rate limits
            time.sleep(0.5)

            return df

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return pd.DataFrame()

    def download_historical(self, symbol: str, start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
        """
        Download news data year by year

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_year : int
            Starting year
        end_year : int
            Ending year

        Returns:
        --------
        pd.DataFrame with all news articles
        """
        print(f"\n{'='*70}")
        print(f"Downloading {symbol} News: {start_year} - {end_year}")
        print(f"{'='*70}")

        all_news = []

        for year in range(start_year, end_year + 1):
            print(f"\n[Year {year}]")

            # Determine end date
            if year == 2025:
                end_date = '2025-10-08'
            else:
                end_date = f'{year}-12-31'

            start_date = f'{year}-01-01'

            # Download batch
            df = self.download_news_batch(symbol, start_date, end_date)

            if len(df) > 0:
                all_news.append(df)

        # Combine all batches
        if all_news:
            combined_df = pd.concat(all_news, ignore_index=True)
            combined_df.sort_values('datetime', inplace=True)

            print(f"\n{'='*70}")
            print(f"Total Articles: {len(combined_df)}")
            print(f"Date Range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            print(f"Unique Dates: {combined_df['date'].nunique()}")
            print(f"{'='*70}")

            return combined_df
        else:
            print("\n⚠ No news data found")
            return pd.DataFrame()

    def save_news(self, df: pd.DataFrame, symbol: str):
        """Save news data to CSV"""
        if len(df) > 0:
            filename = self.output_dir / f"{symbol}_finnhub_historical.csv"
            df.to_csv(filename, index=False)
            print(f"\n✅ Saved to: {filename}")
            return filename
        return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FINNHUB HISTORICAL NEWS DOWNLOAD")
    print("="*70)

    # Initialize downloader
    downloader = FinnhubHistoricalDownloader(output_dir='../01-data')

    # Download Apple (AAPL) news
    print("\n[1/2] APPLE (AAPL)")
    aapl_df = downloader.download_historical('AAPL', start_year=2020, end_year=2025)
    aapl_file = downloader.save_news(aapl_df, 'AAPL')

    # Download Tesla (TSLA) news
    print("\n[2/2] TESLA (TSLA)")
    tsla_df = downloader.download_historical('TSLA', start_year=2020, end_year=2025)
    tsla_file = downloader.save_news(tsla_df, 'TSLA')

    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    if aapl_file:
        print(f"✅ AAPL: {len(aapl_df)} articles → {aapl_file}")
    if tsla_file:
        print(f"✅ TSLA: {len(tsla_df)} articles → {tsla_file}")
    print("="*70 + "\n")
