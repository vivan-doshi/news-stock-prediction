"""
Check which stocks from 50-stock config are missing news data
"""

from pathlib import Path

# 50 stocks (5 per sector)
EXPANDED_STOCKS = [
    # Technology
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL',
    # Finance
    'JPM', 'GS', 'BAC', 'WFC', 'MS',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY',
    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE',
    # Consumer Staples
    'PG', 'WMT', 'KO', 'PEP', 'COST',
    # Communication Services
    'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Industrials
    'BA', 'CAT', 'UPS', 'HON', 'GE',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP',
    # Real Estate
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG',
]

DATA_DIR = Path("../01-data")

existing = []
missing = []

for ticker in EXPANDED_STOCKS:
    news_file = DATA_DIR / f"{ticker}_eodhd_news.csv"
    if news_file.exists():
        existing.append(ticker)
    else:
        missing.append(ticker)

print(f"Total stocks: {len(EXPANDED_STOCKS)}")
print(f"With news data: {len(existing)}")
print(f"Missing news data: {len(missing)}")
print(f"\nMissing: {', '.join(sorted(missing))}")
