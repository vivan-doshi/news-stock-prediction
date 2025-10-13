"""
EXPANDED 50-STOCK CONFIGURATION
================================

5 stocks per sector across 10 sectors = 50 stocks total

Selection criteria:
- Market cap leaders in each sector
- Good liquidity (high trading volume)
- Diverse business models within sector
"""

EXPANDED_STOCKS = {
    # Technology (5)
    'AAPL': {'sector': 'Technology', 'etf': 'XLK', 'name': 'Apple'},
    'MSFT': {'sector': 'Technology', 'etf': 'XLK', 'name': 'Microsoft'},
    'NVDA': {'sector': 'Technology', 'etf': 'XLK', 'name': 'NVIDIA'},
    'AVGO': {'sector': 'Technology', 'etf': 'XLK', 'name': 'Broadcom'},
    'ORCL': {'sector': 'Technology', 'etf': 'XLK', 'name': 'Oracle'},

    # Finance (5)
    'JPM': {'sector': 'Finance', 'etf': 'XLF', 'name': 'JPMorgan Chase'},
    'GS': {'sector': 'Finance', 'etf': 'XLF', 'name': 'Goldman Sachs'},
    'BAC': {'sector': 'Finance', 'etf': 'XLF', 'name': 'Bank of America'},
    'WFC': {'sector': 'Finance', 'etf': 'XLF', 'name': 'Wells Fargo'},
    'MS': {'sector': 'Finance', 'etf': 'XLF', 'name': 'Morgan Stanley'},

    # Healthcare (5)
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV', 'name': 'Johnson & Johnson'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV', 'name': 'Pfizer'},
    'UNH': {'sector': 'Healthcare', 'etf': 'XLV', 'name': 'UnitedHealth'},
    'ABBV': {'sector': 'Healthcare', 'etf': 'XLV', 'name': 'AbbVie'},
    'LLY': {'sector': 'Healthcare', 'etf': 'XLV', 'name': 'Eli Lilly'},

    # Consumer Discretionary (5)
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'name': 'Amazon'},
    'TSLA': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'name': 'Tesla'},
    'HD': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'name': 'Home Depot'},
    'MCD': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'name': 'McDonald\'s'},
    'NKE': {'sector': 'Consumer Discretionary', 'etf': 'XLY', 'name': 'Nike'},

    # Consumer Staples (5)
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP', 'name': 'Procter & Gamble'},
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP', 'name': 'Walmart'},
    'KO': {'sector': 'Consumer Staples', 'etf': 'XLP', 'name': 'Coca-Cola'},
    'PEP': {'sector': 'Consumer Staples', 'etf': 'XLP', 'name': 'PepsiCo'},
    'COST': {'sector': 'Consumer Staples', 'etf': 'XLP', 'name': 'Costco'},

    # Communication Services (5)
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC', 'name': 'Alphabet'},
    'META': {'sector': 'Communication Services', 'etf': 'XLC', 'name': 'Meta'},
    'DIS': {'sector': 'Communication Services', 'etf': 'XLC', 'name': 'Disney'},
    'NFLX': {'sector': 'Communication Services', 'etf': 'XLC', 'name': 'Netflix'},
    'CMCSA': {'sector': 'Communication Services', 'etf': 'XLC', 'name': 'Comcast'},

    # Energy (5)
    'XOM': {'sector': 'Energy', 'etf': 'XLE', 'name': 'ExxonMobil'},
    'CVX': {'sector': 'Energy', 'etf': 'XLE', 'name': 'Chevron'},
    'COP': {'sector': 'Energy', 'etf': 'XLE', 'name': 'ConocoPhillips'},
    'SLB': {'sector': 'Energy', 'etf': 'XLE', 'name': 'Schlumberger'},
    'EOG': {'sector': 'Energy', 'etf': 'XLE', 'name': 'EOG Resources'},

    # Industrials (5)
    'BA': {'sector': 'Industrials', 'etf': 'XLI', 'name': 'Boeing'},
    'CAT': {'sector': 'Industrials', 'etf': 'XLI', 'name': 'Caterpillar'},
    'UPS': {'sector': 'Industrials', 'etf': 'XLI', 'name': 'UPS'},
    'HON': {'sector': 'Industrials', 'etf': 'XLI', 'name': 'Honeywell'},
    'GE': {'sector': 'Industrials', 'etf': 'XLI', 'name': 'General Electric'},

    # Utilities (5)
    'NEE': {'sector': 'Utilities', 'etf': 'XLU', 'name': 'NextEra Energy'},
    'DUK': {'sector': 'Utilities', 'etf': 'XLU', 'name': 'Duke Energy'},
    'SO': {'sector': 'Utilities', 'etf': 'XLU', 'name': 'Southern Company'},
    'D': {'sector': 'Utilities', 'etf': 'XLU', 'name': 'Dominion Energy'},
    'AEP': {'sector': 'Utilities', 'etf': 'XLU', 'name': 'American Electric'},

    # Real Estate (5)
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE', 'name': 'American Tower'},
    'PLD': {'sector': 'Real Estate', 'etf': 'XLRE', 'name': 'Prologis'},
    'CCI': {'sector': 'Real Estate', 'etf': 'XLRE', 'name': 'Crown Castle'},
    'EQIX': {'sector': 'Real Estate', 'etf': 'XLRE', 'name': 'Equinix'},
    'SPG': {'sector': 'Real Estate', 'etf': 'XLRE', 'name': 'Simon Property'},
}

# Summary
def print_summary():
    sectors = {}
    for ticker, info in EXPANDED_STOCKS.items():
        sector = info['sector']
        if sector not in sectors:
            sectors[sector] = []
        sectors[sector].append(ticker)

    print("="*80)
    print("EXPANDED 50-STOCK CONFIGURATION")
    print("="*80)
    print(f"\nTotal stocks: {len(EXPANDED_STOCKS)}")
    print(f"Total sectors: {len(sectors)}\n")

    for sector, tickers in sorted(sectors.items()):
        print(f"{sector}: {len(tickers)} stocks")
        print(f"  {', '.join(tickers)}")
    print()

if __name__ == "__main__":
    print_summary()
