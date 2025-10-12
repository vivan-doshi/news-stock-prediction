"""
Extended test to check how far back Finnhub news data goes
Tests multiple date ranges to find the maximum available history
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')

print("=" * 70)
print("FINNHUB EXTENDED HISTORICAL NEWS TEST")
print("=" * 70)

def test_date_range(ticker, days_back):
    """Test a specific date range"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')

    url = f"https://finnhub.io/api/v1/company-news"
    params = {
        'symbol': ticker,
        'from': from_date,
        'to': to_date,
        'token': FINNHUB_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            if news_data and len(news_data) > 0:
                dates = []
                for article in news_data:
                    if 'datetime' in article:
                        dates.append(datetime.fromtimestamp(article['datetime']))

                if dates:
                    dates.sort()
                    return {
                        'days_back': days_back,
                        'count': len(news_data),
                        'oldest': dates[0],
                        'newest': dates[-1],
                        'span_days': (dates[-1] - dates[0]).days + 1
                    }
    except:
        pass
    return None

# Test different time ranges
print("\nTesting TSLA news availability for different time periods...\n")
test_periods = [7, 14, 30, 60, 90, 180, 365, 730]  # days

results = []
for days in test_periods:
    result = test_date_range('TSLA', days)
    if result:
        results.append(result)
        print(f"Last {days:4d} days: {result['count']:4d} articles | "
              f"Actual range: {result['oldest'].strftime('%Y-%m-%d')} to {result['newest'].strftime('%Y-%m-%d')} "
              f"({result['span_days']} days)")
    else:
        print(f"Last {days:4d} days: No data")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if results:
    max_result = max(results, key=lambda x: x['span_days'])

    print(f"\nMaximum historical data available:")
    print(f"  • Query period: {max_result['days_back']} days back from today")
    print(f"  • Total articles: {max_result['count']}")
    print(f"  • Actual date range: {max_result['oldest'].strftime('%Y-%m-%d')} to {max_result['newest'].strftime('%Y-%m-%d')}")
    print(f"  • Days covered: {max_result['span_days']} days")

    if max_result['span_days'] < 30:
        print(f"\n⚠ WARNING: Only {max_result['span_days']} days of data available")
        print("  This is insufficient for a robust event study analysis")
        print("  Recommendations:")
        print("    1. Use FMP API with paid subscription")
        print("    2. Use alternative news sources (NewsAPI, Alpha Vantage)")
        print("    3. Consider web scraping financial news sites")
        print("    4. Use SEC filings or press releases")
    else:
        print(f"\n✓ {max_result['span_days']} days of data should be sufficient")
        print(f"  Recommended approach:")
        print(f"    • Estimation window: First {max_result['span_days']-20} days")
        print(f"    • Event window: Last 20 days with news events")
else:
    print("\n✗ No historical data available")

print("\n" + "=" * 70)
