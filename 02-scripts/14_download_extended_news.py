"""
DOWNLOAD NEWS FOR EXTENDED STOCKS
==================================

Downloads news for 9 additional stocks (AAPL already downloaded)
"""

import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time

# Load API key
load_dotenv('../.env')
EODHD_API_KEY = os.getenv('EODHD_API_KEY')

if not EODHD_API_KEY:
    raise ValueError("EODHD_API_KEY not found in .env file")

# Stocks needing news (AAPL already has news)
NEW_TICKERS = ['MSFT', 'GS', 'JNJ', 'WMT', 'PG', 'META', 'GOOGL', 'NEE', 'AMT']

START_DATE = '2019-01-01'
END_DATE = '2024-12-31'
DATA_DIR = Path('../01-data')


def download_news(ticker: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """Download news from EODHD API using pagination"""
    print(f"  Downloading news for {ticker}...")

    url = "https://eodhd.com/api/news"
    ticker_symbol = f"{ticker}.US"

    all_articles = []
    offset = 0
    batch_size = 1000

    while True:
        params = {
            's': ticker_symbol,
            'from': start_date,
            'to': end_date,
            'offset': offset,
            'limit': batch_size,
            'api_token': api_key,
            'fmt': 'json'
        }

        try:
            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                data = response.json()

                if not data or len(data) == 0:
                    break

                for article in data:
                    if article is None:
                        continue

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

                print(f"    Batch {offset // batch_size + 1}: {len(data)} articles", end='')
                if len(data) < batch_size:
                    print(" (final)")
                else:
                    print()

                if len(data) < batch_size:
                    break

                offset += batch_size
                time.sleep(0.5)

            elif response.status_code == 429:
                print(f"    Rate limit! Waiting 60 seconds...")
                time.sleep(60)
                continue
            elif response.status_code == 402:
                print(f"    ⚠ Payment required")
                break
            else:
                print(f"    ❌ Error: Status {response.status_code}")
                return None

        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None

    if all_articles:
        df = pd.DataFrame(all_articles)
        df['date'] = pd.to_datetime(df['date'])

        print(f"    ✓ Total: {len(df)} articles")
        print(f"    ✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")

        if 'sentiment_polarity' in df.columns:
            extreme_count = len(df[abs(df['sentiment_polarity']) > 0.95])
            print(f"    ✓ Extreme sentiment (|pol| > 0.95): {extreme_count} ({extreme_count/len(df)*100:.1f}%)")

        return df
    else:
        print(f"    ⚠ No news found")
        return None


def main():
    print("=" * 80)
    print("DOWNLOADING NEWS FOR EXTENDED STOCKS")
    print("=" * 80)
    print(f"\nDate Range: {START_DATE} to {END_DATE}")
    print(f"Tickers: {len(NEW_TICKERS)}")
    print(f"Note: AAPL already downloaded\n")

    results = {'success': [], 'failed': []}
    stats = []

    for i, ticker in enumerate(NEW_TICKERS, 1):
        print(f"[{i}/{len(NEW_TICKERS)}] {ticker}")

        df = download_news(ticker, START_DATE, END_DATE, EODHD_API_KEY)

        if df is not None and not df.empty:
            output_path = DATA_DIR / f"{ticker}_eodhd_news.csv"
            df.to_csv(output_path, index=False)
            print(f"    ✓ Saved to {output_path.name}")

            results['success'].append(ticker)
            stats.append({
                'Ticker': ticker,
                'Articles': len(df),
                'Start': df['date'].min().date(),
                'End': df['date'].max().date()
            })
        else:
            results['failed'].append(ticker)

        if i < len(NEW_TICKERS):
            print("    Waiting 2 seconds...")
            time.sleep(2)

        print()

    # Summary
    print("=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"✓ Successful: {len(results['success'])}/{len(NEW_TICKERS)}")
    if results['success']:
        print(f"  {', '.join(results['success'])}")

    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        print(f"  {', '.join(results['failed'])}")

    if stats:
        print("\n" + "=" * 80)
        print("NEWS DATA SUMMARY")
        print("=" * 80)
        df_stats = pd.DataFrame(stats)
        print(df_stats.to_string(index=False))


if __name__ == "__main__":
    main()
