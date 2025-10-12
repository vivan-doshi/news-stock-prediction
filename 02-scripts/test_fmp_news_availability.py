"""
Test script to check news data availability from Financial Modeling Prep API
Checks the date range of available news for Tesla (TSLA) and Apple (AAPL)
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Get API key
FMP_API_KEY = os.getenv('FNP_API_KEY')

if not FMP_API_KEY:
    raise ValueError("FNP_API_KEY not found in .env file")

print("=" * 70)
print("FMP NEWS DATA AVAILABILITY TEST")
print("=" * 70)

def check_news_availability(ticker, limit=1000):
    """
    Check news availability for a given ticker

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    limit : int
        Maximum number of news articles to fetch (FMP allows up to 1000)

    Returns:
    --------
    dict with news statistics
    """
    print(f"\n[Checking {ticker}]")

    # Try multiple FMP endpoints
    endpoints = [
        f"https://financialmodelingprep.com/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={FMP_API_KEY}",
        f"https://financialmodelingprep.com/api/v4/stock_news?symbol={ticker}&limit={limit}&apikey={FMP_API_KEY}",
        f"https://financialmodelingprep.com/api/v3/fmp/articles?symbol={ticker}&limit={limit}&apikey={FMP_API_KEY}"
    ]

    news_data = None
    working_endpoint = None

    for idx, url in enumerate(endpoints):
        try:
            print(f"  Trying endpoint {idx+1}/3...")
            response = requests.get(url)
            print(f"    Status code: {response.status_code}")
            response.raise_for_status()
            news_data = response.json()

            if isinstance(news_data, dict) and 'error' in news_data:
                print(f"    API Error: {news_data.get('error')}")
                continue

            if news_data and len(news_data) > 0:
                working_endpoint = url
                print(f"  ✓ Successfully connected!")
                break
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            if idx == len(endpoints) - 1:
                print(f"  ✗ All endpoints failed")
            continue

    if not news_data or len(news_data) == 0:
        print(f"  ✗ No news data available")
        return None

    # Parse dates
    dates = []
    for article in news_data:
        if 'publishedDate' in article:
            try:
                date = datetime.fromisoformat(article['publishedDate'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                continue

    if not dates:
        print(f"  ✗ No valid dates found in news data")
        return None

    # Calculate statistics
    dates.sort()
    oldest_date = dates[0]
    newest_date = dates[-1]
    total_articles = len(dates)
    days_coverage = (newest_date - oldest_date).days + 1

    # Calculate articles per day
    avg_articles_per_day = total_articles / days_coverage if days_coverage > 0 else 0

    stats = {
        'ticker': ticker,
        'total_articles': total_articles,
        'oldest_date': oldest_date,
        'newest_date': newest_date,
        'days_coverage': days_coverage,
        'avg_articles_per_day': avg_articles_per_day
    }

    print(f"  ✓ Total articles: {total_articles}")
    print(f"  ✓ Date range: {oldest_date.strftime('%Y-%m-%d')} to {newest_date.strftime('%Y-%m-%d')}")
    print(f"  ✓ Days covered: {days_coverage} days")
    print(f"  ✓ Average articles/day: {avg_articles_per_day:.2f}")

    # Show sample article
    if news_data:
        print(f"\n  Sample article:")
        sample = news_data[0]
        print(f"    Title: {sample.get('title', 'N/A')[:80]}...")
        print(f"    Date: {sample.get('publishedDate', 'N/A')}")
        print(f"    Site: {sample.get('site', 'N/A')}")

    return stats

# Test both tickers
print(f"\nUsing API key: {FMP_API_KEY[:10]}...")
print(f"Fetching up to 1000 most recent articles for each ticker\n")

# Check Tesla
tesla_stats = check_news_availability('TSLA', limit=1000)

# Check Apple
apple_stats = check_news_availability('AAPL', limit=1000)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if tesla_stats and apple_stats:
    print(f"\n{tesla_stats['ticker']}:")
    print(f"  • {tesla_stats['total_articles']} articles")
    print(f"  • {tesla_stats['days_coverage']} days ({tesla_stats['oldest_date'].strftime('%Y-%m-%d')} to {tesla_stats['newest_date'].strftime('%Y-%m-%d')})")
    print(f"  • {tesla_stats['avg_articles_per_day']:.2f} articles/day")

    print(f"\n{apple_stats['ticker']}:")
    print(f"  • {apple_stats['total_articles']} articles")
    print(f"  • {apple_stats['days_coverage']} days ({apple_stats['oldest_date'].strftime('%Y-%m-%d')} to {apple_stats['newest_date'].strftime('%Y-%m-%d')})")
    print(f"  • {apple_stats['avg_articles_per_day']:.2f} articles/day")

    # Find common date range
    common_start = max(tesla_stats['oldest_date'], apple_stats['oldest_date'])
    common_end = min(tesla_stats['newest_date'], apple_stats['newest_date'])
    common_days = (common_end - common_start).days + 1

    print(f"\nCommon date range:")
    print(f"  • {common_start.strftime('%Y-%m-%d')} to {common_end.strftime('%Y-%m-%d')}")
    print(f"  • {common_days} days")

    print(f"\n{'✓' if common_days > 30 else '✗'} Data {'is' if common_days > 30 else 'may not be'} sufficient for analysis")
    print(f"  (Recommended: at least 30 days for event study)")

else:
    print("\n✗ Could not retrieve data for both tickers")
    print("  Please check your API key and internet connection")

print("\n" + "=" * 70)
