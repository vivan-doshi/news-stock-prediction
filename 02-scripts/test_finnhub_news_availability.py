"""
Test script to check news data availability from Finnhub API
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
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

if not FINNHUB_API_KEY:
    raise ValueError("FINNHUB_API_KEY not found in .env file")

print("=" * 70)
print("FINNHUB NEWS DATA AVAILABILITY TEST")
print("=" * 70)

def check_news_availability(ticker, days_back=365):
    """
    Check news availability for a given ticker using Finnhub

    Finnhub free tier allows:
    - Company news from the past year
    - Limited to ~60 API calls per minute

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        How many days back to check (max 365 for free tier)

    Returns:
    --------
    dict with news statistics
    """
    print(f"\n[Checking {ticker}]")

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Finnhub uses YYYY-MM-DD format
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    print(f"  Querying from {from_date} to {to_date}")

    # Finnhub company news endpoint
    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': from_date,
        'to': to_date,
        'token': FINNHUB_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        print(f"  Status: {response.status_code}")

        if response.status_code == 401:
            print("  ✗ Authentication failed. Please check your Finnhub API key.")
            return None
        elif response.status_code == 403:
            print("  ✗ Access forbidden. This may require a paid plan.")
            return None
        elif response.status_code == 429:
            print("  ✗ Rate limit exceeded. Please wait and try again.")
            return None

        response.raise_for_status()
        news_data = response.json()

        if not news_data or len(news_data) == 0:
            print(f"  ✗ No news data found for {ticker}")
            return None

        # Parse dates
        dates = []
        for article in news_data:
            if 'datetime' in article:
                try:
                    # Finnhub returns Unix timestamp
                    date = datetime.fromtimestamp(article['datetime'])
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
            print(f"    Title: {sample.get('headline', 'N/A')[:80]}...")
            if 'datetime' in sample:
                sample_date = datetime.fromtimestamp(sample['datetime'])
                print(f"    Date: {sample_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Source: {sample.get('source', 'N/A')}")
            print(f"    URL: {sample.get('url', 'N/A')[:60]}...")

        return stats

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None

# Test API key validity first
print(f"\nTesting API key: {FINNHUB_API_KEY[:10]}...")
test_url = f"https://finnhub.io/api/v1/quote?symbol=AAPL&token={FINNHUB_API_KEY}"
try:
    response = requests.get(test_url)
    print(f"API Key Test Status: {response.status_code}")
    if response.status_code == 200:
        print("✓ API key is valid")
    elif response.status_code == 401:
        print("✗ API key is invalid")
        exit(1)
    elif response.status_code == 403:
        print("⚠ API key valid but has limited access")
except Exception as e:
    print(f"✗ Error testing API key: {e}")

# Check Tesla
print("\n" + "-" * 70)
tesla_stats = check_news_availability('TSLA', days_back=365)

# Check Apple
print("\n" + "-" * 70)
apple_stats = check_news_availability('AAPL', days_back=365)

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

    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if tesla_stats['total_articles'] < 10 or apple_stats['total_articles'] < 10:
        print("  ⚠ Low article count - consider using a longer time period or multiple sources")
    if tesla_stats['avg_articles_per_day'] < 0.5 or apple_stats['avg_articles_per_day'] < 0.5:
        print("  ⚠ Low article frequency - event study may have sparse data")
    if common_days >= 30:
        print("  ✓ Sufficient data for event study analysis")
        print(f"  ✓ Recommended event window: [-10, +10] days around news events")

elif tesla_stats or apple_stats:
    print("\n⚠ Partial data available:")
    for stats in [tesla_stats, apple_stats]:
        if stats:
            print(f"\n{stats['ticker']}:")
            print(f"  • {stats['total_articles']} articles")
            print(f"  • {stats['days_coverage']} days ({stats['oldest_date'].strftime('%Y-%m-%d')} to {stats['newest_date'].strftime('%Y-%m-%d')})")

else:
    print("\n✗ Could not retrieve data for either ticker")
    print("\nPossible solutions:")
    print("  1. Verify Finnhub API key is correct")
    print("  2. Check if free tier has access to company news")
    print("  3. Consider using alternative news sources")

print("\n" + "=" * 70)
print("NOTE: Finnhub free tier limitations:")
print("  • Company news limited to past 12 months")
print("  • 60 API calls per minute")
print("  • May have fewer articles than paid tiers")
print("=" * 70)
