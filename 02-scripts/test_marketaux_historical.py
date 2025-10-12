"""
Test script to check historical data availability in Marketaux API
Tests different date ranges to see how far back data goes
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')

if not MARKETAUX_API_KEY:
    print("Error: MARKETAUX_API_KEY not found in .env file")
    exit(1)

print("=" * 70)
print("MARKETAUX HISTORICAL DATA TEST")
print("=" * 70)

def test_date_range(ticker, days_back, published_on=None):
    """
    Test a specific date range

    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    days_back : int
        How many days back from today
    published_on : str, optional
        Specific date to query (YYYY-MM-DD format)
    """

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        'symbols': ticker,
        'filter_entities': 'true',
        'language': 'en',
        'api_token': MARKETAUX_API_KEY,
        'limit': 100
    }

    # Add date parameter if specified
    if published_on:
        params['published_on'] = published_on
        date_str = published_on
    else:
        date_str = f"{days_back} days ago"

    try:
        response = requests.get(url, params=params)

        if response.status_code != 200:
            return None, f"Error {response.status_code}"

        data = response.json()

        if 'data' not in data or not data['data']:
            return None, "No data"

        news_data = data['data']

        # Parse dates
        dates = []
        for article in news_data:
            if 'published_at' in article:
                try:
                    date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                    dates.append(date)
                except:
                    continue

        if dates:
            dates.sort()
            result = {
                'count': len(news_data),
                'oldest': dates[0],
                'newest': dates[-1],
                'span_days': (dates[-1] - dates[0]).days + 1
            }

            # Get rate limit info
            if 'meta' in data:
                result['remaining'] = data['meta'].get('remaining', 'Unknown')
                result['limit'] = data['meta'].get('limit', 'Unknown')

            return result, None
        else:
            return None, "No valid dates"

    except Exception as e:
        return None, str(e)

print(f"\nAPI Key: {MARKETAUX_API_KEY[:10]}...")
print("\nTesting historical data availability for TSLA...\n")

# Test 1: Check without date filter (should get recent data)
print("[Test 1] Querying without date filter (default recent news)")
result, error = test_date_range('TSLA', 0)
if result:
    print(f"  ✓ Found {result['count']} articles")
    print(f"  ✓ Date range: {result['oldest'].strftime('%Y-%m-%d')} to {result['newest'].strftime('%Y-%m-%d')}")
    print(f"  ✓ Span: {result['span_days']} days")
    print(f"  ✓ Remaining API calls: {result.get('remaining', 'Unknown')}/{result.get('limit', 'Unknown')}")
    most_recent = result['newest']
    oldest_available = result['oldest']
else:
    print(f"  ✗ {error}")
    most_recent = datetime.now()
    oldest_available = datetime.now()

time.sleep(1)  # Be nice to the API

# Test 2: Check specific historical dates
print("\n[Test 2] Testing specific historical dates...")
test_dates = [
    (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),   # 1 day ago
    (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),   # 3 days ago
    (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),   # 7 days ago
    (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),  # 14 days ago
    (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),  # 30 days ago
    (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),  # 60 days ago
]

available_dates = []
for test_date in test_dates:
    print(f"\n  Testing date: {test_date}")
    result, error = test_date_range('TSLA', 0, published_on=test_date)

    if result:
        print(f"    ✓ Found {result['count']} articles")
        print(f"    ✓ Date range: {result['oldest'].strftime('%Y-%m-%d')} to {result['newest'].strftime('%Y-%m-%d')}")
        available_dates.append(test_date)
        print(f"    ✓ Remaining calls: {result.get('remaining', 'Unknown')}")
    else:
        print(f"    ✗ {error}")

    time.sleep(1)  # Rate limiting

# Test 3: Check using published_after parameter
print("\n[Test 3] Testing published_after parameter...")
after_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
print(f"\n  Querying news after {after_date}...")

url = "https://api.marketaux.com/v1/news/all"
params = {
    'symbols': 'TSLA',
    'filter_entities': 'true',
    'published_after': after_date,
    'language': 'en',
    'api_token': MARKETAUX_API_KEY,
    'limit': 100
}

try:
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get('data'):
            dates = []
            for article in data['data']:
                if 'published_at' in article:
                    try:
                        date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                        dates.append(date)
                    except:
                        continue

            if dates:
                dates.sort()
                print(f"  ✓ Found {len(data['data'])} articles")
                print(f"  ✓ Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
                print(f"  ✓ Span: {(dates[-1] - dates[0]).days + 1} days")

                if 'meta' in data:
                    print(f"  ✓ Remaining calls: {data['meta'].get('remaining', 'Unknown')}/{data['meta'].get('limit', 'Unknown')}")
        else:
            print("  ✗ No data returned")
    else:
        print(f"  ✗ Error {response.status_code}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if available_dates:
    print(f"\n✓ Historical data is available!")
    print(f"  Oldest date with data: {min(available_dates)}")
    print(f"  Newest date with data: {max(available_dates)}")
    print(f"  Total days tested with data: {len(available_dates)}")

    oldest = datetime.strptime(min(available_dates), '%Y-%m-%d')
    newest = datetime.strptime(max(available_dates), '%Y-%m-%d')
    span = (newest - oldest).days + 1

    print(f"\n  Estimated historical coverage: ~{span} days")

    if span >= 30:
        print(f"\n  ✓ {span} days is sufficient for event study!")
        print(f"    Recommended: Use 20-25 days for estimation window")
        print(f"    Recommended: Use 5-10 days for event window")
    else:
        print(f"\n  ⚠ Only {span} days may be limited for traditional event study")
        print(f"    Consider: Short-term event study or intraday analysis")
else:
    print("\n✗ No historical data found in tested date ranges")
    print("  Free tier may only include very recent news (last 1-3 days)")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print("\nBased on the tests, to get comprehensive historical data:")
print("  1. If free tier has 30+ days: Proceed with event study")
print("  2. If free tier has <30 days: Consider upgrading or using Finnhub")
print("  3. Finnhub currently provides 8 days with high article volume")
print("  4. Could combine both APIs to maximize coverage")
print("=" * 70)
