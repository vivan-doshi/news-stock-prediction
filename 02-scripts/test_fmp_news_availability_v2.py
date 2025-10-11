"""
Test script to check news data availability from Financial Modeling Prep API
Checks the date range of available news for Tesla (TSLA) and Apple (AAPL)
Using FMP General News endpoints
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
print("FMP NEWS DATA AVAILABILITY TEST (V2)")
print("=" * 70)

def check_news_availability(ticker, page_limit=10):
    """
    Check news availability for a given ticker using FMP's general news endpoint

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    page_limit : int
        Maximum number of pages to fetch (each page has ~50-100 articles)

    Returns:
    --------
    dict with news statistics
    """
    print(f"\n[Checking {ticker}]")

    all_articles = []

    # Try the general news endpoint which works with free tier
    # This endpoint returns general market news
    for page in range(page_limit):
        url = f"https://financialmodelingprep.com/api/v3/fmp/articles?page={page}&size=100&apikey={FMP_API_KEY}"

        try:
            print(f"  Fetching page {page+1}/{page_limit}...")
            response = requests.get(url)
            print(f"    Status: {response.status_code}")

            if response.status_code == 403:
                print(f"  ✗ 403 Forbidden - This endpoint requires paid subscription")
                break

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and 'error' in data:
                print(f"    API Error: {data.get('error')}")
                break

            if isinstance(data, dict) and 'content' in data:
                articles = data['content']
            elif isinstance(data, list):
                articles = data
            else:
                print(f"    Unexpected response format")
                break

            if not articles or len(articles) == 0:
                print(f"    No more articles")
                break

            # Filter articles mentioning the ticker
            ticker_articles = [a for a in articles if ticker.upper() in str(a).upper()]
            all_articles.extend(ticker_articles)
            print(f"    Found {len(ticker_articles)} articles mentioning {ticker}")

        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
            break

    if not all_articles:
        # Try company-specific press releases endpoint
        print(f"  Trying press releases endpoint...")
        url = f"https://financialmodelingprep.com/api/v3/press-releases/{ticker}?page=0&apikey={FMP_API_KEY}"

        try:
            response = requests.get(url)
            print(f"    Status: {response.status_code}")
            response.raise_for_status()
            press_data = response.json()

            if press_data and isinstance(press_data, list) and len(press_data) > 0:
                all_articles = press_data
                print(f"  ✓ Found {len(all_articles)} press releases")
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")

    if not all_articles:
        print(f"  ✗ No news data available for {ticker}")
        return None

    # Parse dates
    dates = []
    for article in all_articles:
        date_field = article.get('publishedDate') or article.get('date') or article.get('datetime')
        if date_field:
            try:
                date = datetime.fromisoformat(str(date_field).replace('Z', '+00:00'))
                dates.append(date)
            except:
                try:
                    date = datetime.strptime(str(date_field)[:10], '%Y-%m-%d')
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
    if all_articles:
        print(f"\n  Sample article:")
        sample = all_articles[0]
        title = sample.get('title') or sample.get('text', '')
        print(f"    Title: {title[:80]}...")
        date_field = sample.get('publishedDate') or sample.get('date') or sample.get('datetime')
        print(f"    Date: {date_field}")

    return stats

# Test API key validity first
print(f"\nTesting API key: {FMP_API_KEY[:10]}...")
test_url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={FMP_API_KEY}"
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
tesla_stats = check_news_availability('TSLA', page_limit=5)

# Check Apple
print("\n" + "-" * 70)
apple_stats = check_news_availability('AAPL', page_limit=5)

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

elif tesla_stats or apple_stats:
    print("\n⚠ Partial data available:")
    for stats in [tesla_stats, apple_stats]:
        if stats:
            print(f"\n{stats['ticker']}:")
            print(f"  • {stats['total_articles']} articles")
            print(f"  • {stats['days_coverage']} days ({stats['oldest_date'].strftime('%Y-%m-%d')} to {stats['newest_date'].strftime('%Y-%m-%d')})")

else:
    print("\n✗ Could not retrieve data for either ticker")
    print("\nPossible reasons:")
    print("  1. Free tier API key has limited access to news endpoints")
    print("  2. News data requires a paid FMP subscription")
    print("  3. Try using alternative news sources (Finnhub, News API, etc.)")

print("\n" + "=" * 70)
