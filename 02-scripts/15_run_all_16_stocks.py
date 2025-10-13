"""
RUN COMPREHENSIVE 16-STOCK ANALYSIS
====================================

Simple runner script that uses the existing analysis module
"""

import sys
import os

# Set working directory to scripts folder
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the analysis module
import importlib
multi_sector_module = importlib.import_module('11_multi_sector_deviation_analysis')
MultiSectorDeviationAnalysis = multi_sector_module.MultiSectorDeviationAnalysis

from pathlib import Path

# All 16 stocks
COMPREHENSIVE_STOCK_MAP = {
    # Original 6
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'},

    # Additional 10
    'MSFT': {'sector': 'Technology', 'etf': 'XLK'},
    'AAPL': {'sector': 'Technology', 'etf': 'XLK'},
    'GS': {'sector': 'Finance', 'etf': 'XLF'},
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV'},
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'META': {'sector': 'Communication Services', 'etf': 'XLC'},
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC'},
    'NEE': {'sector': 'Utilities', 'etf': 'XLU'},
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE'}
}

OUTPUT_DIR = Path('../03-output/results/comprehensive_16_stock')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    """Run comprehensive 16-stock analysis"""
    print("=" * 80)
    print("COMPREHENSIVE 16-STOCK DEVIATION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Stocks: {len(COMPREHENSIVE_STOCK_MAP)}")

    from collections import defaultdict
    by_sector = defaultdict(list)
    for stock, info in COMPREHENSIVE_STOCK_MAP.items():
        by_sector[info['sector']].append(stock)

    print(f"Sectors: {len(by_sector)}\n")

    print("Stock Distribution:")
    for sector in sorted(by_sector.keys()):
        print(f"  {sector:25}: {', '.join(sorted(by_sector[sector]))}")
    print()

    # Create analyzer
    analyzer = MultiSectorDeviationAnalysis(
        stock_sector_map=COMPREHENSIVE_STOCK_MAP,
        polarity_threshold=0.95
    )

    # Override output directory
    analyzer.output_dir = OUTPUT_DIR

    # Run analysis
    results_df = analyzer.run_multi_sector_analysis()

    # Additional analysis
    if results_df is not None and len(results_df) > 0:
        import pandas as pd
        import numpy as np

        print("\n" + "=" * 80)
        print("SECTOR-LEVEL ANALYSIS")
        print("=" * 80)

        # Group by sector
        sector_stats = results_df.groupby('sector').agg({
            'deviation_increase_pct': ['mean', 'median', 'min', 'max', 'count'],
            'significant': 'sum',
            'news_days': 'mean',
            'p_value': 'median'
        }).round(3)

        print("\nAggregate by Sector:")
        print(sector_stats.to_string())

        sector_stats.to_csv(OUTPUT_DIR / 'sector_level_summary.csv')

        # Technology deep dive
        tech_stocks = results_df[results_df['sector'] == 'Technology']
        if len(tech_stocks) > 0:
            print("\n" + "-" * 80)
            print("TECHNOLOGY SECTOR DETAIL (3 stocks)")
            print("-" * 80)
            print(tech_stocks[['ticker', 'news_days', 'deviation_increase_pct', 'p_value', 'significant']].to_string(index=False))

    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE 16-STOCK ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults: {OUTPUT_DIR}")

    return results_df


if __name__ == "__main__":
    main()
