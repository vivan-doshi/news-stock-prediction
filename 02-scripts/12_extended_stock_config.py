"""
EXTENDED STOCK CONFIGURATION
=============================

Adding 10 more stocks for comprehensive multi-sector analysis.

Total stocks: 16 across 11 sectors
"""

# Original 6 stocks
ORIGINAL_STOCKS = {
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'}
}

# 10 Additional stocks for broader coverage
ADDITIONAL_STOCKS = {
    # Technology (add 2 more)
    'MSFT': {'sector': 'Technology', 'etf': 'XLK'},
    'AAPL': {'sector': 'Technology', 'etf': 'XLK'},

    # Finance (add 1 more)
    'GS': {'sector': 'Finance', 'etf': 'XLF'},

    # Healthcare (add 1 more - biotech)
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV'},

    # Consumer Staples (new sector)
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP'},

    # Communication Services (new sector)
    'META': {'sector': 'Communication Services', 'etf': 'XLC'},
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC'},

    # Utilities (new sector - typically low volatility)
    'NEE': {'sector': 'Utilities', 'etf': 'XLU'},

    # Real Estate (new sector)
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE'}
}

# Combined mapping
EXTENDED_STOCK_SECTOR_MAP = {**ORIGINAL_STOCKS, **ADDITIONAL_STOCKS}

# Sector ETFs we'll need to download
SECTOR_ETFS = list(set([info['etf'] for info in EXTENDED_STOCK_SECTOR_MAP.values()]))

print("=" * 80)
print("EXTENDED STOCK CONFIGURATION")
print("=" * 80)
print(f"\nOriginal Stocks: {len(ORIGINAL_STOCKS)}")
print(f"Additional Stocks: {len(ADDITIONAL_STOCKS)}")
print(f"Total Stocks: {len(EXTENDED_STOCK_SECTOR_MAP)}")
print(f"\nSectors Covered: {len(set([info['sector'] for info in EXTENDED_STOCK_SECTOR_MAP.values()]))}")
print(f"Sector ETFs: {len(SECTOR_ETFS)}")

print("\n" + "=" * 80)
print("STOCK BREAKDOWN BY SECTOR")
print("=" * 80)

from collections import defaultdict
by_sector = defaultdict(list)
for stock, info in EXTENDED_STOCK_SECTOR_MAP.items():
    by_sector[info['sector']].append(stock)

for sector in sorted(by_sector.keys()):
    stocks = ', '.join(sorted(by_sector[sector]))
    etf = EXTENDED_STOCK_SECTOR_MAP[by_sector[sector][0]]['etf']
    print(f"{sector:25} ({etf}): {stocks}")

print("\n" + "=" * 80)
print("NEW DOWNLOADS NEEDED")
print("=" * 80)
print(f"\nStocks: {', '.join(sorted(ADDITIONAL_STOCKS.keys()))}")
print(f"ETFs: {', '.join(sorted(set([ADDITIONAL_STOCKS[s]['etf'] for s in ADDITIONAL_STOCKS]) - set([ORIGINAL_STOCKS[s]['etf'] for s in ORIGINAL_STOCKS])))}")
