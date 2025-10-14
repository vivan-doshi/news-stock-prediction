"""
Test EODHD API to check if news scarcity in 2019-2020 is API limitation or actual data
"""
import os
import requests
from dotenv import load_dotenv
from pathlib import Path
import time

# Load API key
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
api_key = os.getenv('EODHD_API_KEY')

# Test API for AAPL in different years
test_periods = [
    ('2019-01-01', '2019-12-31', '2019'),
    ('2020-01-01', '2020-12-31', '2020'),
    ('2021-01-01', '2021-12-31', '2021'),
    ('2022-01-01', '2022-12-31', '2022'),
    ('2023-01-01', '2023-12-31', '2023'),
    ('2024-01-01', '2024-12-31', '2024'),
]

print('='*80)
print('Testing EODHD API for AAPL news availability by year')
print('='*80)
print()
print(f"{'Year':<8} {'Articles':<12} {'First Date':<30} {'Last Date':<30}")
print('-' * 80)

for start, end, year in test_periods:
    params = {
        's': 'AAPL.US',
        'from': start,
        'to': end,
        'offset': 0,
        'limit': 1000,
        'api_token': api_key,
        'fmt': 'json'
    }

    try:
        response = requests.get('https://eodhd.com/api/news', params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            count = len(data) if data else 0

            first_date = 'N/A'
            last_date = 'N/A'

            if data and len(data) > 0:
                first_date = data[0].get('date', 'N/A')[:19]
                last_date = data[-1].get('date', 'N/A')[:19]

            print(f"{year:<8} {count:<12} {first_date:<30} {last_date:<30}")

            # Show first article title for context
            if count > 0:
                print(f"         Sample: {data[0].get('title', 'No title')[:60]}...")
        else:
            print(f"{year:<8} ERROR: HTTP {response.status_code}")
    except Exception as e:
        print(f"{year:<8} ERROR: {str(e)[:50]}")

    time.sleep(0.5)  # Rate limiting

print()
print('='*80)
print('Analysis:')
print('- If API returns few articles for 2019-2020, it\'s an API data limitation')
print('- If API returns many articles but we have few, it\'s a download script issue')
print('='*80)
