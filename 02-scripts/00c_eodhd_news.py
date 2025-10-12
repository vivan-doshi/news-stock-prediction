"""
EODHD News Data Acquisition Module
Downloads historical news data from EODHD API for event study analysis
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import time
import json

class EODHDNewsAcquisition:
    """Downloads news data from EODHD API"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "../01-data"):
        """
        Initialize EODHD news downloader

        Parameters:
        -----------
        api_key : str, optional
            EODHD API key (will load from .env if not provided)
        output_dir : str
            Directory to save news data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load API key
        if api_key is None:
            env_path = Path(__file__).parent.parent / '.env'
            load_dotenv(env_path)
            api_key = os.getenv('EODHD_API_KEY')

        if not api_key:
            raise ValueError(
                "EODHD_API_KEY not found. Add it to .env file.\n"
                "Get your API key at: https://eodhd.com/register"
            )

        self.api_key = api_key
        self.base_url = "https://eodhd.com/api/news"

    def download_news(self,
                     ticker: str,
                     start_date: str,
                     end_date: Optional[str] = None,
                     limit: int = 1000,
                     save: bool = True) -> pd.DataFrame:
        """
        Download news articles for a ticker

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL' for Apple)
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str, optional
            End date (default: today)
        limit : int
            Max articles per request (default: 1000, max: 1000)
        save : bool
            Save to CSV

        Returns:
        --------
        pd.DataFrame with news articles
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"\nDownloading news for {ticker} from {start_date} to {end_date}...")

        # EODHD requires ticker format like "AAPL.US"
        ticker_symbol = f"{ticker}.US"

        all_articles = []
        offset = 0
        batch_size = limit

        while True:
            params = {
                's': ticker_symbol,
                'from': start_date,
                'to': end_date,
                'offset': offset,
                'limit': batch_size,
                'api_token': self.api_key,
                'fmt': 'json'
            }

            try:
                print(f"  Fetching batch {offset // batch_size + 1} (offset: {offset})...", end=" ")
                response = requests.get(self.base_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if not data or len(data) == 0:
                        print("No more data")
                        break

                    for article in data:
                        # Skip if article is None
                        if article is None:
                            continue

                        # Safely get sentiment data
                        sentiment = article.get('sentiment', {})
                        if sentiment is None:
                            sentiment = {}

                        all_articles.append({
                            'ticker': ticker,
                            'date': article.get('date'),
                            'title': article.get('title', ''),
                            'content': article.get('content', ''),
                            'link': article.get('link', ''),
                            'symbols': article.get('symbols', ''),
                            'tags': article.get('tags', ''),
                            'sentiment_polarity': sentiment.get('polarity', None),
                            'sentiment_neg': sentiment.get('neg', None),
                            'sentiment_neu': sentiment.get('neu', None),
                            'sentiment_pos': sentiment.get('pos', None)
                        })

                    print(f"{len(data)} articles")

                    # Check if we got less than limit (means we're done)
                    if len(data) < batch_size:
                        break

                    offset += batch_size
                    time.sleep(0.5)  # Rate limiting

                elif response.status_code == 429:
                    print("Rate limit!")
                    print("    Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                elif response.status_code == 402:
                    print("Payment required - upgrade plan for historical data")
                    break
                else:
                    print(f"Error: Status {response.status_code}")
                    print(f"    Response: {response.text[:200]}")
                    break

            except requests.exceptions.Timeout:
                print("Timeout! Try narrowing date range")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['date'] = pd.to_datetime(df['date'])

            print(f"\n  ✓ Total: {len(all_articles)} articles")
            print(f"  ✓ Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  ✓ Unique dates: {df['date'].dt.date.nunique()}")
        else:
            print(f"\n  ⚠ No news found for {ticker}")
            df = pd.DataFrame(columns=['ticker', 'date', 'title', 'content', 'link',
                                      'symbols', 'tags', 'sentiment_polarity',
                                      'sentiment_neg', 'sentiment_neu', 'sentiment_pos'])

        if save and not df.empty:
            filename = self.output_dir / f"{ticker}_eodhd_news.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ Saved to {filename}")

        return df


if __name__ == "__main__":
    print("="*70)
    print("EODHD News Acquisition")
    print("="*70)
    print("\nNote: Free plan = 20 API calls/day (4 news requests)")
    print("      Each news request costs 5 API calls")
    print("      For full historical data, consider paid plan ($19.99/month)")

    # Initialize downloader
    try:
        downloader = EODHDNewsAcquisition(output_dir='../01-data')

        # Download news for Apple (AAPL)
        print("\n[1/2] APPLE (AAPL)")
        print("-"*70)
        aapl_news = downloader.download_news(
            ticker='AAPL',
            start_date='2020-01-01',
            end_date='2024-01-31',
            save=True
        )

        # Download news for Tesla (TSLA)
        print("\n[2/2] TESLA (TSLA)")
        print("-"*70)
        tsla_news = downloader.download_news(
            ticker='TSLA',
            start_date='2020-01-01',
            end_date='2025-10-08',
            save=True
        )

        # Summary
        print("\n" + "="*70)
        print("Download Summary")
        print("="*70)
        if not aapl_news.empty:
            print(f"  AAPL: {len(aapl_news)} articles")
        if not tsla_news.empty:
            print(f"  TSLA: {len(tsla_news)} articles")
        print("="*70)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nSetup Instructions:")
        print("  1. Register at https://eodhd.com/register")
        print("  2. Get your API key from dashboard")
        print("  3. Add to .env file: EODHD_API_KEY=your_key_here")
