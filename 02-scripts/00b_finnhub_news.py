"""
Finnhub News Data Acquisition Module
Downloads company news data from Finnhub API
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import time
import json
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')


class FinnhubNewsAcquisition:
    """Handles news data download from Finnhub API"""

    def __init__(self, api_key: Optional[str] = None, output_dir: str = "../01-data"):
        """
        Parameters:
        -----------
        api_key : str, optional
            Finnhub API key. If not provided, will try to load from .env file
        output_dir : str
            Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load API key
        if api_key is None:
            load_dotenv()
            api_key = os.getenv('FINNHUB_API_KEY')

        if api_key is None:
            raise ValueError(
                "Finnhub API key not found. Please either:\n"
                "1. Pass api_key parameter, or\n"
                "2. Set FINNHUB_API_KEY in .env file, or\n"
                "3. Set FINNHUB_API_KEY environment variable\n"
                "Get your free API key at: https://finnhub.io/register"
            )

        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"

    def get_company_news(self,
                        symbol: str,
                        start_date: str,
                        end_date: Optional[str] = None,
                        save: bool = True) -> pd.DataFrame:
        """
        Get company news from Finnhub

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        save : bool
            Save to CSV file (default: True)

        Returns:
        --------
        pd.DataFrame with news data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading {symbol} news from {start_date} to {end_date}...")

        # Finnhub API endpoint
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
                print(f"  ⚠ No news found for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(news_data)

            # Convert Unix timestamp to datetime
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df['date'] = df['datetime'].dt.date
            df['date'] = pd.to_datetime(df['date'])

            # Sort by date
            df.sort_values('datetime', inplace=True)

            print(f"  ✓ Downloaded {len(df)} news articles")

            if save:
                filename = self.output_dir / f"{symbol}_news_raw.csv"
                df.to_csv(filename, index=False)
                print(f"  ✓ Saved raw news to {filename}")

            return df

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Error downloading news: {e}")
            raise

    def extract_news_dates(self,
                          news_df: pd.DataFrame,
                          min_articles_per_day: int = 1,
                          save: bool = True) -> pd.DataFrame:
        """
        Extract unique news dates from news data

        Parameters:
        -----------
        news_df : pd.DataFrame
            DataFrame with news data
        min_articles_per_day : int
            Minimum number of articles required to count as a news day
        save : bool
            Save to CSV file

        Returns:
        --------
        pd.DataFrame with news dates and article counts
        """
        if len(news_df) == 0:
            return pd.DataFrame(columns=['Date', 'Article_Count'])

        # Count articles per day
        date_counts = news_df.groupby('date').size().reset_index(name='Article_Count')
        date_counts.rename(columns={'date': 'Date'}, inplace=True)

        # Filter by minimum articles
        date_counts = date_counts[date_counts['Article_Count'] >= min_articles_per_day]

        print(f"  ✓ Extracted {len(date_counts)} unique news dates")
        print(f"    (min {min_articles_per_day} article(s) per day)")

        if save:
            filename = self.output_dir / "news_dates.csv"
            date_counts.to_csv(filename, index=False)
            print(f"  ✓ Saved news dates to {filename}")

        return date_counts

    def get_news_with_sentiment(self,
                                news_df: pd.DataFrame,
                                save: bool = True) -> pd.DataFrame:
        """
        Process news data and extract sentiment indicators

        Parameters:
        -----------
        news_df : pd.DataFrame
            Raw news data
        save : bool
            Save processed data

        Returns:
        --------
        pd.DataFrame with processed news and sentiment
        """
        if len(news_df) == 0:
            return pd.DataFrame()

        print("Processing news sentiment...")

        # Create processed DataFrame
        processed = news_df[['date', 'datetime', 'headline', 'source', 'url', 'summary']].copy()

        # Simple sentiment indicators based on keywords
        # (This is basic - Phase 2 will use proper NLP models)
        positive_keywords = [
            'beats', 'surge', 'soars', 'record', 'profit', 'growth', 'gain',
            'up', 'rise', 'rally', 'bullish', 'positive', 'strong', 'robust'
        ]

        negative_keywords = [
            'miss', 'falls', 'drops', 'loss', 'decline', 'down', 'plunge',
            'weak', 'bearish', 'negative', 'concern', 'risk', 'warning'
        ]

        def simple_sentiment(headline):
            """Simple keyword-based sentiment"""
            if pd.isna(headline):
                return 'neutral'

            headline_lower = headline.lower()

            pos_count = sum(1 for word in positive_keywords if word in headline_lower)
            neg_count = sum(1 for word in negative_keywords if word in headline_lower)

            if pos_count > neg_count:
                return 'positive'
            elif neg_count > pos_count:
                return 'negative'
            else:
                return 'neutral'

        processed['simple_sentiment'] = processed['headline'].apply(simple_sentiment)

        print(f"  ✓ Processed {len(processed)} articles")
        print(f"    Positive: {(processed['simple_sentiment'] == 'positive').sum()}")
        print(f"    Negative: {(processed['simple_sentiment'] == 'negative').sum()}")
        print(f"    Neutral: {(processed['simple_sentiment'] == 'neutral').sum()}")

        if save:
            filename = self.output_dir / "news_processed.csv"
            processed.to_csv(filename, index=False)
            print(f"  ✓ Saved processed news to {filename}")

        return processed

    def get_major_news_dates(self,
                            news_df: pd.DataFrame,
                            min_articles: int = 3,
                            keywords: Optional[List[str]] = None,
                            save: bool = True) -> pd.DataFrame:
        """
        Extract major news dates based on criteria

        Parameters:
        -----------
        news_df : pd.DataFrame
            News data
        min_articles : int
            Minimum articles to qualify as major news
        keywords : List[str], optional
            Keywords indicating major news (e.g., ['earnings', 'acquisition'])
        save : bool
            Save major news dates

        Returns:
        --------
        pd.DataFrame with major news dates
        """
        if len(news_df) == 0:
            return pd.DataFrame()

        print(f"Identifying major news dates...")

        # Count articles per day
        daily_counts = news_df.groupby('date').agg({
            'headline': [('Article_Count', 'count'), ('Sample_Headlines', lambda x: ' | '.join(x[:3]))],
            'source': [('Sources', lambda x: list(x.unique()))]
        }).reset_index()

        # Flatten multi-level columns
        daily_counts.columns = ['Date', 'Article_Count', 'Sample_Headlines', 'Sources']

        # Filter for major news
        major_news = daily_counts[daily_counts['Article_Count'] >= min_articles]

        # If keywords provided, also filter by keywords
        if keywords:
            keyword_pattern = '|'.join(keywords)
            major_news = major_news[
                major_news['Sample_Headlines'].str.contains(keyword_pattern, case=False, na=False)
            ]

        print(f"  ✓ Identified {len(major_news)} major news dates")
        print(f"    (min {min_articles} articles per day)")

        if save:
            filename = self.output_dir / "major_news_dates.csv"
            major_news.to_csv(filename, index=False)
            print(f"  ✓ Saved major news dates to {filename}")

        return major_news

    def download_all_news_data(self,
                               symbol: str,
                               start_date: str,
                               end_date: Optional[str] = None,
                               min_articles_per_day: int = 1) -> Dict[str, pd.DataFrame]:
        """
        Download and process all news data in one go

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        start_date : str
            Start date
        end_date : str, optional
            End date
        min_articles_per_day : int
            Minimum articles to count as news day

        Returns:
        --------
        dict with all processed news DataFrames
        """
        print("=" * 70)
        print("FINNHUB NEWS DATA ACQUISITION")
        print("=" * 70)

        data = {}

        # 1. Get raw news
        print("\n[1/4] Downloading company news...")
        data['raw_news'] = self.get_company_news(symbol, start_date, end_date)

        if len(data['raw_news']) == 0:
            print("\n⚠ No news data available")
            return data

        # 2. Extract news dates
        print("\n[2/4] Extracting news dates...")
        data['news_dates'] = self.extract_news_dates(
            data['raw_news'],
            min_articles_per_day=min_articles_per_day
        )

        # 3. Process sentiment
        print("\n[3/4] Processing sentiment...")
        data['processed_news'] = self.get_news_with_sentiment(data['raw_news'])

        # 4. Identify major news
        print("\n[4/4] Identifying major news dates...")
        data['major_news'] = self.get_major_news_dates(data['raw_news'], min_articles=3)

        print("\n" + "=" * 70)
        print("NEWS DATA ACQUISITION COMPLETE!")
        print("=" * 70)
        print(f"\nFiles saved to: {self.output_dir.absolute()}")

        return data


def download_finnhub_news(symbol: str = 'AAPL',
                          start_date: str = '2020-01-01',
                          end_date: Optional[str] = None,
                          api_key: Optional[str] = None,
                          output_dir: str = '../01-data'):
    """
    Convenience function to download Finnhub news data

    Example usage:
    --------------
    download_finnhub_news('AAPL', '2020-01-01', api_key='your_api_key')
    """
    acquirer = FinnhubNewsAcquisition(api_key=api_key, output_dir=output_dir)
    return acquirer.download_all_news_data(symbol, start_date, end_date)


if __name__ == "__main__":
    print("Finnhub News Data Acquisition Module")
    print("\nSetup:")
    print("  1. Get free API key at: https://finnhub.io/register")
    print("  2. Add to .env file: FINNHUB_API_KEY=your_key_here")
    print("\nExample usage:")
    print("  from finnhub_news import download_finnhub_news")
    print("  download_finnhub_news('AAPL', '2020-01-01')")
    print("\nNote: Free tier has rate limits (60 calls/minute)")