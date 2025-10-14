"""
Test EODHD API pagination to see total available articles per year
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

# Test pagination for each year
test_periods = [
    ('2019-01-01', '2019-12-31', '2019'),
    ('2020-01-01', '2020-12-31', '2020'),
    ('2021-01-01', '2021-12-31', '2021'),
]

print('='*80)
print('Testing EODHD API pagination - Total available articles per year')
print('='*80)
print()

for start, end, year in test_periods:
    print(f"\n{year}:")
    print('-' * 40)

    total_articles = 0
    offset = 0
    batch = 1

    while True:
        params = {
            's': 'AAPL.US',
            'from': start,
            'to': end,
            'offset': offset,
            'limit': 1000,
            'api_token': api_key,
            'fmt': 'json'
        }

        try:
            response = requests.get('https://eodhd.com/api/news', params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                count = len(data) if data else 0

                if count == 0:
                    break

                total_articles += count
                print(f"  Batch {batch}: {count} articles (offset {offset})")

                # Check if we got less than limit, meaning no more data
                if count < 1000:
                    break

                offset += 1000
                batch += 1
                time.sleep(0.5)  # Rate limiting

            else:
                print(f"  ERROR: HTTP {response.status_code}")
                break

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            break

    print(f"\n  TOTAL for {year}: {total_articles} articles")
    print(f"  Downloaded in our data: {[302, 302, 7512][int(year)-2019] if int(year) <= 2021 else 'N/A'}")

print()
print('='*80)
print('Conclusion:')
print('This shows whether EODHD truly has limited data for 2019-2020')
print('or if our download script stopped early.')
print('='*80)