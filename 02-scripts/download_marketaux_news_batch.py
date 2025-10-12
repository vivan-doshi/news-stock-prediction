"""
Marketaux News Acquisition - Batch Download
Downloads news data efficiently using date ranges
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import time

class MarketauxBatchDownloader:
    """Downloads news data from Marketaux API efficiently"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "../01-data"):
        """Initialize Marketaux batch downloader"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load API key
        if api_key is None:
            env_path = Path(__file__).parent.parent / '.env'
            load_dotenv(env_path)
            api_key = os.getenv('MARKETAUX_API_KEY')

        if not api_key:
            raise ValueError("MARKETAUX_API_KEY not found. Add it to .env file.")

        self.api_key = api_key
        self.base_url = "https://api.marketaux.com/v1/news/all"

    def download_news_batch(self,
                           ticker: str,
                           start_date: str,
                           end_date: str,
                           batch_days: int = 30) -> pd.DataFrame:
        """
        Download news articles in batches

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str
            End date 'YYYY-MM-DD'
        batch_days : int
            Number of days per batch (default: 30)

        Returns:
        --------
        pd.DataFrame with all news articles
        """
        print(f"\nDownloading news for {ticker} from {start_date} to {end_date}")
        print(f"Using {batch_days}-day batches...")

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_articles = []
        current_start = start
        batch_num = 0

        while current_start <= end:
            current_end = min(current_start + timedelta(days=batch_days - 1), end)
            batch_num += 1

            params = {
                'symbols': ticker,
                'filter_entities': 'true',
                'published_after': current_start.strftime('%Y-%m-%dT00:00:00'),
                'published_before': current_end.strftime('%Y-%m-%dT23:59:59'),
                'language': 'en',
                'api_token': self.api_key,
                'limit': 100
            }

            try:
                print(f"  Batch {batch_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...", end=" ")
                response = requests.get(self.base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', [])

                    for article in articles:
                        all_articles.append({
                            'ticker': ticker,
                            'published_at': article.get('published_at'),
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'snippet': article.get('snippet', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', ''),
                            'entities': ','.join([e.get('symbol', '') for e in article.get('entities', [])])
                        })

                    print(f"{len(articles)} articles")

                    # Check pagination
                    meta = data.get('meta', {})
                    if meta.get('found', 0) > 100:
                        print(f"    Note: Found {meta.get('found')} articles, but API limit is 100 per request")

                elif response.status_code == 429:
                    print("Rate limit!")
                    print("    Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    print(f"Error: Status {response.status_code}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"Error: {e}")

            current_start = current_end + timedelta(days=1)

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['published_at'] = pd.to_datetime(df['published_at'])
            df['date'] = df['published_at'].dt.date

            print(f"\n  ✓ Total: {len(all_articles)} articles across {df['date'].nunique()} days")
        else:
            print(f"\n  ⚠ No news found for {ticker}")
            df = pd.DataFrame(columns=['ticker', 'published_at', 'title', 'description',
                                      'snippet', 'url', 'source', 'entities', 'date'])

        return df

    def download_and_save(self, ticker: str, start_date: str, end_date: str):
        """Download and save news for a ticker"""
        df = self.download_news_batch(ticker, start_date, end_date)

        if not df.empty:
            filename = self.output_dir / f"{ticker}_marketaux_news.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ Saved to {filename}")
            return df
        return None


if __name__ == "__main__":
    print("="*70)
    print("Marketaux News Batch Download")
    print("="*70)

    # Initialize downloader
    downloader = MarketauxBatchDownloader(output_dir='../01-data')

    # Download news for Apple (AAPL)
    print("\n[1/2] APPLE (AAPL)")
    print("-"*70)
    aapl_news = downloader.download_and_save(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2025-10-08'
    )

    # Download news for Tesla (TSLA)
    print("\n[2/2] TESLA (TSLA)")
    print("-"*70)
    tsla_news = downloader.download_and_save(
        ticker='TSLA',
        start_date='2020-01-01',
        end_date='2025-10-08'
    )

    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    if aapl_news is not None:
        print(f"  AAPL: {len(aapl_news)} articles")
    if tsla_news is not None:
        print(f"  TSLA: {len(tsla_news)} articles")
    print("="*70)
