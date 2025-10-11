"""
Test script to check news data availability from Marketaux API
Marketaux provides financial news with free tier access
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
MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')

if not MARKETAUX_API_KEY:
    print("=" * 70)
    print("MARKETAUX API KEY NOT FOUND")
    print("=" * 70)
    print("\nPlease add your Marketaux API key to the .env file:")
    print("1. Get free API key at: https://www.marketaux.com/")
    print("2. Add to .env file: MARKETAUX_API_KEY=your_key_here")
    print("\nFree tier includes:")
    print("  • 100 API calls per day")
    print("  • News from the past 7 days")
    print("  • Multiple filters and symbols")
    exit(1)

print("=" * 70)
print("MARKETAUX NEWS DATA AVAILABILITY TEST")
print("=" * 70)

def check_news_availability(ticker, days_back=7):
    """
    Check news availability for a given ticker using Marketaux

    Marketaux free tier:
    - 100 API calls per day
    - News from the past 7 days
    - Filters by symbols, entities, industries

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        How many days back to check (max 7 for free tier)

    Returns:
    --------
    dict with news statistics
    """
    print(f"\n[Checking {ticker}]")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(days_back, 7))  # Free tier: max 7 days

    # Marketaux API endpoint
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        'symbols': ticker,
        'filter_entities': 'true',
        'language': 'en',
        'api_token': MARKETAUX_API_KEY,
        'limit': 100  # Max results per request
    }

    try:
        print(f"  Querying Marketaux API...")
        response = requests.get(url, params=params)
        print(f"  Status: {response.status_code}")

        if response.status_code == 401:
            print("  ✗ Authentication failed. Please check your Marketaux API key.")
            return None
        elif response.status_code == 403:
            print("  ✗ Access forbidden. This may require a paid plan.")
            return None
        elif response.status_code == 429:
            print("  ✗ Rate limit exceeded (100 calls/day). Please wait.")
            return None

        response.raise_for_status()
        data = response.json()

        if 'data' not in data or not data['data']:
            print(f"  ✗ No news data found for {ticker}")
            return None

        news_data = data['data']
        print(f"  ✓ Retrieved {len(news_data)} articles")

        # Parse dates
        dates = []
        for article in news_data:
            if 'published_at' in article:
                try:
                    date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
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
            print(f"    Date: {sample.get('published_at', 'N/A')}")
            print(f"    Source: {sample.get('source', 'N/A')}")
            print(f"    Sentiment: {sample.get('entities', [{}])[0].get('sentiment_score', 'N/A') if sample.get('entities') else 'N/A'}")

        # Show rate limit info
        if 'meta' in data:
            meta = data['meta']
            print(f"\n  Rate limit info:")
            if 'remaining' in meta:
                print(f"    Remaining calls today: {meta['remaining']}")
            if 'limit' in meta:
                print(f"    Daily limit: {meta['limit']}")

        return stats

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None

# Test API key validity first
print(f"\nTesting API key: {MARKETAUX_API_KEY[:10]}...")

# Check Tesla
print("\n" + "-" * 70)
tesla_stats = check_news_availability('TSLA', days_back=7)

# Check Apple
print("\n" + "-" * 70)
apple_stats = check_news_availability('AAPL', days_back=7)

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

    print(f"\n{'✓' if common_days >= 7 else '✗'} Data {'is' if common_days >= 7 else 'may not be'} available")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"  ⚠ Marketaux free tier only provides {common_days} days of historical data")
    print(f"  ⚠ This is insufficient for a traditional event study (need 60+ days)")
    print(f"\n  Options:")
    print(f"    1. Upgrade to Marketaux paid plan ($19/month)")
    print(f"       - 30+ days of historical data")
    print(f"       - 10,000 API calls per month")
    print(f"    2. Combine multiple free APIs to extend coverage")
    print(f"    3. Use web scraping for historical news")
    print(f"    4. Use alternative approach with available data")

elif tesla_stats or apple_stats:
    print("\n⚠ Partial data available:")
    for stats in [tesla_stats, apple_stats]:
        if stats:
            print(f"\n{stats['ticker']}:")
            print(f"  • {stats['total_articles']} articles")
            print(f"  • {stats['days_coverage']} days ({stats['oldest_date'].strftime('%Y-%m-%d')} to {stats['newest_date'].strftime('%Y-%m-%d')})")

else:
    print("\n✗ Could not retrieve data for either ticker")

print("\n" + "=" * 70)
print("MARKETAUX FREE TIER LIMITATIONS:")
print("  • 100 API calls per day")
print("  • News from the past 7 days only")
print("  • No sentiment scores on free tier (requires paid plan)")
print("=" * 70)
