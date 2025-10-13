"""
EXPANDED STOCK CONFIGURATION
=============================

Minimum 5 stocks per sector for robust experimentation.
No duplicate stocks across sectors.

Target: 40+ stocks across 8 major sectors
"""

from collections import defaultdict

# Comprehensive stock configuration with at least 5 stocks per sector
EXPANDED_STOCK_CONFIG = {
    # TECHNOLOGY (XLK) - 7 stocks
    'AAPL': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Apple - Consumer Electronics'},
    'MSFT': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Microsoft - Software'},
    'NVDA': {'sector': 'Technology', 'etf': 'XLK', 'description': 'NVIDIA - Semiconductors'},
    'AVGO': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Broadcom - Semiconductors'},
    'ORCL': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Oracle - Enterprise Software'},
    'CRM': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Salesforce - Cloud/SaaS'},
    'ADBE': {'sector': 'Technology', 'etf': 'XLK', 'description': 'Adobe - Creative Software'},

    # FINANCE (XLF) - 6 stocks
    'JPM': {'sector': 'Finance', 'etf': 'XLF', 'description': 'JPMorgan Chase - Banking'},
    'BAC': {'sector': 'Finance', 'etf': 'XLF', 'description': 'Bank of America - Banking'},
    'WFC': {'sector': 'Finance', 'etf': 'XLF', 'description': 'Wells Fargo - Banking'},
    'GS': {'sector': 'Finance', 'etf': 'XLF', 'description': 'Goldman Sachs - Investment Banking'},
    'MS': {'sector': 'Finance', 'etf': 'XLF', 'description': 'Morgan Stanley - Investment Banking'},
    'BLK': {'sector': 'Finance', 'etf': 'XLF', 'description': 'BlackRock - Asset Management'},

    # HEALTHCARE (XLV) - 6 stocks
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'Johnson & Johnson - Pharmaceuticals'},
    'UNH': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'UnitedHealth - Health Insurance'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'Pfizer - Pharmaceuticals'},
    'ABBV': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'AbbVie - Biopharmaceuticals'},
    'TMO': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'Thermo Fisher - Life Sciences'},
    'LLY': {'sector': 'Healthcare', 'etf': 'XLV', 'description': 'Eli Lilly - Pharmaceuticals'},

    # CONSUMER DISCRETIONARY (XLY) - 5 stocks
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'description': 'Amazon - E-commerce/Cloud'},
    'TSLA': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'description': 'Tesla - Electric Vehicles'},
    'HD': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'description': 'Home Depot - Home Improvement'},
    'MCD': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'description': 'McDonald\'s - Fast Food'},
    'NKE': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'description': 'Nike - Apparel'},

    # CONSUMER STAPLES (XLP) - 5 stocks
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP', 'description': 'Procter & Gamble - Consumer Products'},
    'KO': {'sector': 'Consumer Staples', 'etf': 'XLP', 'description': 'Coca-Cola - Beverages'},
    'PEP': {'sector': 'Consumer Staples', 'etf': 'XLP', 'description': 'PepsiCo - Food & Beverages'},
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP', 'description': 'Walmart - Retail'},
    'COST': {'sector': 'Consumer Staples', 'etf': 'XLP', 'description': 'Costco - Wholesale Retail'},

    # COMMUNICATION SERVICES (XLC) - 5 stocks
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC', 'description': 'Alphabet - Internet Services'},
    'META': {'sector': 'Communication Services', 'etf': 'XLC', 'description': 'Meta - Social Media'},
    'NFLX': {'sector': 'Communication Services', 'etf': 'XLC', 'description': 'Netflix - Streaming'},
    'DIS': {'sector': 'Communication Services', 'etf': 'XLC', 'description': 'Disney - Entertainment'},
    'CMCSA': {'sector': 'Communication Services', 'etf': 'XLC', 'description': 'Comcast - Telecom/Media'},

    # INDUSTRIALS (XLI) - 5 stocks
    'BA': {'sector': 'Industrials', 'etf': 'XLI', 'description': 'Boeing - Aerospace'},
    'CAT': {'sector': 'Industrials', 'etf': 'XLI', 'description': 'Caterpillar - Heavy Machinery'},
    'UNP': {'sector': 'Industrials', 'etf': 'XLI', 'description': 'Union Pacific - Rail Transport'},
    'HON': {'sector': 'Industrials', 'etf': 'XLI', 'description': 'Honeywell - Conglomerate'},
    'GE': {'sector': 'Industrials', 'etf': 'XLI', 'description': 'General Electric - Industrial'},

    # ENERGY (XLE) - 5 stocks
    'XOM': {'sector': 'Energy', 'etf': 'XLE', 'description': 'Exxon Mobil - Oil & Gas'},
    'CVX': {'sector': 'Energy', 'etf': 'XLE', 'description': 'Chevron - Oil & Gas'},
    'COP': {'sector': 'Energy', 'etf': 'XLE', 'description': 'ConocoPhillips - Oil & Gas'},
    'SLB': {'sector': 'Energy', 'etf': 'XLE', 'description': 'Schlumberger - Oil Services'},
    'EOG': {'sector': 'Energy', 'etf': 'XLE', 'description': 'EOG Resources - Oil & Gas Exploration'},
}

# Additional specialty sectors (optional for extended analysis)
SPECIALTY_SECTORS = {
    # UTILITIES (XLU) - 5 stocks
    'NEE': {'sector': 'Utilities', 'etf': 'XLU', 'description': 'NextEra Energy - Utilities'},
    'DUK': {'sector': 'Utilities', 'etf': 'XLU', 'description': 'Duke Energy - Utilities'},
    'SO': {'sector': 'Utilities', 'etf': 'XLU', 'description': 'Southern Company - Utilities'},
    'D': {'sector': 'Utilities', 'etf': 'XLU', 'description': 'Dominion Energy - Utilities'},
    'AEP': {'sector': 'Utilities', 'etf': 'XLU', 'description': 'American Electric Power - Utilities'},

    # REAL ESTATE (XLRE) - 5 stocks
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE', 'description': 'American Tower - Cell Towers'},
    'PLD': {'sector': 'Real Estate', 'etf': 'XLRE', 'description': 'Prologis - Industrial REITs'},
    'CCI': {'sector': 'Real Estate', 'etf': 'XLRE', 'description': 'Crown Castle - Infrastructure'},
    'EQIX': {'sector': 'Real Estate', 'etf': 'XLRE', 'description': 'Equinix - Data Center REITs'},
    'SPG': {'sector': 'Real Estate', 'etf': 'XLRE', 'description': 'Simon Property - Retail REITs'},
}

# Combine all stocks
ALL_STOCKS = {**EXPANDED_STOCK_CONFIG, **SPECIALTY_SECTORS}

# Get unique sector ETFs
SECTOR_ETFS = sorted(list(set([info['etf'] for info in ALL_STOCKS.values()])))

def print_configuration():
    """Print comprehensive breakdown of stock configuration"""
    print("=" * 100)
    print("EXPANDED STOCK CONFIGURATION FOR EXPERIMENTATION")
    print("=" * 100)

    # Overall stats
    print(f"\nTotal Stocks: {len(ALL_STOCKS)}")
    print(f"Core Stocks (8 major sectors): {len(EXPANDED_STOCK_CONFIG)}")
    print(f"Specialty Stocks (2 additional sectors): {len(SPECIALTY_SECTORS)}")

    # Sector breakdown
    by_sector = defaultdict(list)
    for stock, info in ALL_STOCKS.items():
        by_sector[info['sector']].append((stock, info['description']))

    print(f"\nTotal Sectors: {len(by_sector)}")
    print(f"Sector ETFs: {len(SECTOR_ETFS)}")

    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN BY SECTOR")
    print("=" * 100)

    for sector in sorted(by_sector.keys()):
        stocks = by_sector[sector]
        etf = ALL_STOCKS[stocks[0][0]]['etf']
        print(f"\n{sector} ({etf}) - {len(stocks)} stocks:")
        for stock, desc in sorted(stocks):
            print(f"  • {stock:6} - {desc}")

    print("\n" + "=" * 100)
    print("STOCK SYMBOLS BY SECTOR (for quick reference)")
    print("=" * 100)

    for sector in sorted(by_sector.keys()):
        stocks = by_sector[sector]
        etf = ALL_STOCKS[stocks[0][0]]['etf']
        stock_list = ', '.join([s[0] for s in sorted(stocks)])
        print(f"{sector:25} ({etf:5}): {stock_list}")

    print("\n" + "=" * 100)
    print("ETFs TO DOWNLOAD")
    print("=" * 100)
    print(f"\n{', '.join(SECTOR_ETFS)}")

    # Check for duplicates
    print("\n" + "=" * 100)
    print("VALIDATION")
    print("=" * 100)

    all_tickers = list(ALL_STOCKS.keys())
    if len(all_tickers) == len(set(all_tickers)):
        print("✓ No duplicate stocks found")
    else:
        duplicates = [t for t in all_tickers if all_tickers.count(t) > 1]
        print(f"✗ WARNING: Duplicate stocks found: {set(duplicates)}")

    # Check minimum stocks per sector
    min_stocks = min([len(stocks) for stocks in by_sector.values()])
    if min_stocks >= 5:
        print(f"✓ All sectors have minimum 5 stocks (minimum: {min_stocks})")
    else:
        sectors_below = [s for s, st in by_sector.items() if len(st) < 5]
        print(f"✗ WARNING: Some sectors have < 5 stocks: {sectors_below}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    print_configuration()
