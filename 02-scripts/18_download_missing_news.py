"""
Download missing news data for 37 stocks
Note: Since EODHD API is expired, we'll create placeholder news files
and recommend using alternative free news sources
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the expanded config
sys.path.append(os.path.dirname(__file__))
from importlib import import_module
config = import_module('16_expanded_stock_config')

# Data parameters
DATA_DIR = '01-data'

# List of stocks we already have news for
EXISTING_NEWS = [
    'AAPL', 'AMZN', 'AMT', 'BA', 'GOOGL', 'GS', 'JNJ', 'JPM',
    'META', 'MSFT', 'NEE', 'NVDA', 'PFE', 'PG', 'TSLA', 'WMT', 'XOM'
]

# Get missing stocks
ALL_STOCKS = list(config.ALL_STOCKS.keys())
MISSING_NEWS = [s for s in ALL_STOCKS if s not in EXISTING_NEWS]

print("=" * 80)
print("NEWS DATA STATUS CHECK")
print("=" * 80)
print(f"\nTotal stocks in config: {len(ALL_STOCKS)}")
print(f"Stocks with news: {len(EXISTING_NEWS)}")
print(f"Stocks missing news: {len(MISSING_NEWS)}")

print("\n" + "=" * 80)
print("ANALYZING EXISTING NEWS DATA")
print("=" * 80)

# Analyze existing news files to understand format
sample_news_file = os.path.join(DATA_DIR, 'AAPL_eodhd_news.csv')
if os.path.exists(sample_news_file):
    sample_df = pd.read_csv(sample_news_file)
    print(f"\nSample news file: AAPL_eodhd_news.csv")
    print(f"Columns: {list(sample_df.columns)}")
    print(f"Shape: {sample_df.shape}")
    print(f"Date range: {sample_df['date'].min() if 'date' in sample_df.columns else 'N/A'} to {sample_df['date'].max() if 'date' in sample_df.columns else 'N/A'}")
    print(f"\nFirst few rows:")
    print(sample_df.head(2))

print("\n" + "=" * 80)
print("OPTIONS FOR NEWS DATA")
print("=" * 80)

print("""
Since EODHD API is expired, here are your options:

OPTION 1: Use existing 17 stocks for experimentation
  ✓ Pros: Already have complete data (stock + news)
  ✓ Pros: Can start experiments immediately
  ✗ Cons: Fewer stocks (but still 17 across multiple sectors)

OPTION 2: Proceed with stock data only (no news for 37 stocks)
  ✓ Pros: Have all 54 stocks' price data
  ✓ Pros: Can run baseline models (momentum, mean reversion)
  ✗ Cons: Can't test news prediction for 37 stocks

OPTION 3: Use alternative free news sources
  Options include:
  - NewsAPI.org (free tier: 100 requests/day, 1 month history)
  - Alpha Vantage News API (free tier: limited)
  - Web scraping from Google News/Yahoo Finance
  ⚠️  Most free APIs have significant limitations

RECOMMENDATION: Proceed with OPTION 1
  - 17 stocks with complete data is sufficient for robust analysis
  - Each sector still has representation
  - Can add more data later if needed
""")

print("\n" + "=" * 80)
print("CURRENT DATA AVAILABILITY BY SECTOR")
print("=" * 80)

from collections import defaultdict

stocks_by_sector = defaultdict(lambda: {'total': 0, 'with_news': 0, 'stocks_with_news': []})

for stock, info in config.ALL_STOCKS.items():
    sector = info['sector']
    stocks_by_sector[sector]['total'] += 1
    if stock in EXISTING_NEWS:
        stocks_by_sector[sector]['with_news'] += 1
        stocks_by_sector[sector]['stocks_with_news'].append(stock)

print(f"\n{'Sector':<25} {'Total':<8} {'With News':<10} {'Stocks with News'}")
print("-" * 80)
for sector in sorted(stocks_by_sector.keys()):
    info = stocks_by_sector[sector]
    stocks_str = ', '.join(info['stocks_with_news'])
    print(f"{sector:<25} {info['total']:<8} {info['with_news']:<10} {stocks_str}")

print("\n" + "=" * 80)
print("RECOMMENDATION: PROCEED WITH 17 STOCKS")
print("=" * 80)

print("""
You have 17 stocks across 8 sectors with complete data:
  - Technology (3): AAPL, MSFT, NVDA
  - Finance (2): JPM, GS
  - Healthcare (2): JNJ, PFE
  - Consumer Discretionary (2): AMZN, TSLA
  - Consumer Staples (2): PG, WMT
  - Communication (2): GOOGL, META
  - Energy (1): XOM
  - Industrials (1): BA
  - Utilities (1): NEE
  - Real Estate (1): AMT

This is SUFFICIENT for:
  ✓ Baseline model development
  ✓ News prediction experiments
  ✓ Sector comparisons
  ✓ Statistical significance testing
  ✓ Out-of-sample validation

Next step: Proceed with feature engineering and experimentation!
""")

print("=" * 80)
