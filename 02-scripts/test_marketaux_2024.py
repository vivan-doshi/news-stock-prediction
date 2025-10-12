"""
Test script to check if Marketaux has 2024 historical data
"""

import os
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

MARKETAUX_API_KEY = os.getenv('MARKETAUX_API_KEY')

print("=" * 70)
print("MARKETAUX 2024 HISTORICAL DATA TEST")
print("=" * 70)

def test_specific_date(ticker, date_str):
    """Test a specific date"""
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        'symbols': ticker,
        'filter_entities': 'true',
        'published_on': date_str,
        'language': 'en',
        'api_token': MARKETAUX_API_KEY,
        'limit': 100
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                articles = data['data']
                dates = []
                for article in articles:
                    if 'published_at' in article:
                        try:
                            date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                            dates.append(date)
                        except:
                            continue

                if dates:
                    dates.sort()
                    return {
                        'success': True,
                        'count': len(articles),
                        'oldest': dates[0],
                        'newest': dates[-1]
                    }

        return {'success': False, 'error': f"Status {response.status_code}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}

print(f"\nAPI Key: {MARKETAUX_API_KEY[:10]}...")
print("\nTesting 2024 dates for TSLA...\n")

# Test various 2024 dates
test_dates_2024 = [
    '2024-12-31',  # End of 2024
    '2024-12-01',  # December
    '2024-11-01',  # November
    '2024-10-01',  # October
    '2024-09-01',  # September
    '2024-08-01',  # August
    '2024-07-01',  # July
    '2024-06-01',  # June
    '2024-05-01',  # May
    '2024-04-01',  # April
    '2024-03-01',  # March
    '2024-02-01',  # February
    '2024-01-01',  # January
]

available_2024 = []
earliest_date = None
latest_date = None

for date_str in test_dates_2024:
    print(f"Testing {date_str}...", end=' ')
    result = test_specific_date('TSLA', date_str)

    if result['success']:
        print(f"✓ Found {result['count']} articles ({result['oldest'].strftime('%Y-%m-%d')} to {result['newest'].strftime('%Y-%m-%d')})")
        available_2024.append(date_str)

        if earliest_date is None or result['oldest'] < earliest_date:
            earliest_date = result['oldest']
        if latest_date is None or result['newest'] > latest_date:
            latest_date = result['newest']
    else:
        print(f"✗ {result.get('error', 'No data')}")

    time.sleep(1)  # Be nice to the API

# Test even earlier dates
print("\nTesting 2023 dates...")
test_dates_2023 = ['2023-12-01', '2023-06-01', '2023-01-01']

for date_str in test_dates_2023:
    print(f"Testing {date_str}...", end=' ')
    result = test_specific_date('TSLA', date_str)

    if result['success']:
        print(f"✓ Found {result['count']} articles")
        if earliest_date is None or result['oldest'] < earliest_date:
            earliest_date = result['oldest']
    else:
        print(f"✗ {result.get('error', 'No data')}")

    time.sleep(1)

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if available_2024:
    print(f"\n✓ 2024 historical data IS AVAILABLE!")
    print(f"\n  Months with data in 2024: {len(available_2024)} out of {len(test_dates_2024)} tested")
    print(f"  Dates tested with data:")
    for date in available_2024:
        print(f"    • {date}")

    if earliest_date and latest_date:
        total_span = (latest_date - earliest_date).days
        print(f"\n  Overall date range found:")
        print(f"    Earliest: {earliest_date.strftime('%Y-%m-%d')}")
        print(f"    Latest: {latest_date.strftime('%Y-%m-%d')}")
        print(f"    Total span: ~{total_span} days")

        if total_span >= 365:
            print(f"\n  ✓✓ EXCELLENT: {total_span} days is more than enough for event study!")
        elif total_span >= 90:
            print(f"\n  ✓ GOOD: {total_span} days is sufficient for event study!")
        elif total_span >= 30:
            print(f"\n  ⚠ OK: {total_span} days is minimal for event study")

        print(f"\n  RECOMMENDED EVENT STUDY PARAMETERS:")
        print(f"    • Estimation window: 60-120 days")
        print(f"    • Event window: [-10, +10] days around news")
        print(f"    • Total window needed: ~140 days")
        print(f"    • Available: {total_span} days {'✓' if total_span >= 140 else '✗'}")

else:
    print("\n✗ No 2024 data found")
    print("  Free tier may only include recent data")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)

if available_2024 and len(available_2024) >= 6:
    print("\n✓ Marketaux has sufficient 2024 historical data!")
    print("\n  1. You can proceed with event study using 2024 data")
    print("  2. Download news for key dates in 2024")
    print("  3. Match with stock price data for same period")
    print("  4. Run event study analysis")
else:
    print("\n⚠ Limited 2024 data - consider alternatives:")
    print("  1. Use the most recent 60 days available (2025 data)")
    print("  2. Combine Marketaux + Finnhub for more coverage")
    print("  3. Upgrade to paid tier for full historical access")

print("=" * 70)
