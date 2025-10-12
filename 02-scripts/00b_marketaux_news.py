"""
Marketaux News Acquisition Module
Downloads news data from Marketaux API for event study analysis
"""

import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import time

class MarketauxNewsAcquisition:
    """Downloads news data from Marketaux API"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "../01-data"):
        """
        Initialize Marketaux news downloader

        Parameters:
        -----------
        api_key : str, optional
            Marketaux API key (will load from .env if not provided)
        output_dir : str
            Directory to save news data
        """
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

    def download_news(self,
                     ticker: str,
                     start_date: str,
                     end_date: Optional[str] = None,
                     save: bool = True) -> pd.DataFrame:
        """
        Download news articles for a ticker

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str, optional
            End date (default: today)
        save : bool
            Save to CSV

        Returns:
        --------
        pd.DataFrame with all news articles
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading news for {ticker} from {start_date} to {end_date}...")

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_articles = []
        current_date = start

        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')

            params = {
                'symbols': ticker,
                'filter_entities': 'true',
                'published_on': date_str,
                'language': 'en',
                'api_token': self.api_key,
                'limit': 100
            }

            try:
                response = requests.get(self.base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    articles = data.get('data', [])

                    for article in articles:
                        all_articles.append({
                            'ticker': ticker,
                            'date': date_str,
                            'published_at': article.get('published_at'),
                            'title': article.get('title', ''),
                            'description': article.get('description', ''),
                            'snippet': article.get('snippet', ''),
                            'url': article.get('url', ''),
                            'source': article.get('source', '')
                        })

                    print(f"  {date_str}: {len(articles)} articles")

                elif response.status_code == 429:
                    print(f"  Rate limit reached on {date_str}. Waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    print(f"  Error on {date_str}: Status {response.status_code}")

                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"  Error on {date_str}: {e}")

            current_date += timedelta(days=1)

        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            df['date'] = pd.to_datetime(df['date'])
            df['published_at'] = pd.to_datetime(df['published_at'])

            print(f"  ✓ Found {len(all_articles)} articles across {df['date'].nunique()} days")
        else:
            print(f"  ⚠ No news found for {ticker}")
            df = pd.DataFrame(columns=['ticker', 'date', 'published_at', 'title', 'description', 'snippet', 'url', 'source'])

        if save and not df.empty:
            filename = self.output_dir / f"{ticker}_marketaux_news.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ Saved to {filename}")

        return df

def download_news(ticker: str = 'AAPL',
                 start_date: str = '2024-01-01',
                 end_date: Optional[str] = None,
                 api_key: Optional[str] = None,
                 output_dir: str = '../01-data'):
    """
    Convenience function to download news

    Example:
    --------
    download_news('AAPL', '2024-01-01')
    """
    downloader = MarketauxNewsAcquisition(api_key=api_key, output_dir=output_dir)
    return downloader.download_news(ticker, start_date, end_date)

if __name__ == "__main__":
    print("="*60)
    print("Marketaux News Acquisition")
    print("="*60)

    # Initialize downloader
    downloader = MarketauxNewsAcquisition(output_dir='../01-data')

    # Download news for Apple (AAPL) - Same period as Tesla
    print("\n[1/2] Downloading AAPL news...")
    aapl_news = downloader.download_news(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2025-10-08',
        save=True
    )

    # Download news for Tesla (TSLA)
    print("\n[2/2] Downloading TSLA news...")
    tsla_news = downloader.download_news(
        ticker='TSLA',
        start_date='2020-01-01',
        end_date='2025-10-08',
        save=True
    )

    print("\n" + "="*60)
    print("Summary:")
    print(f"  AAPL: {len(aapl_news)} articles")
    print(f"  TSLA: {len(tsla_news)} articles")
    print("="*60)
