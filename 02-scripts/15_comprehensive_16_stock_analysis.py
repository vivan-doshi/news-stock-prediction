"""
COMPREHENSIVE 16-STOCK DEVIATION ANALYSIS
==========================================

Analyzes news-driven deviations across 16 stocks from 10 different sectors.
"""

import sys
sys.path.insert(0, '.')

# Import the analysis class
from sys import path
path.insert(0, '02-scripts')

import importlib
spec = importlib.util.spec_from_file_location("multi_sector", "11_multi_sector_deviation_analysis.py")
multi_sector_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_sector_module)

MultiSectorDeviationAnalysis = multi_sector_module.MultiSectorDeviationAnalysis

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

from pathlib import Path
OUTPUT_DIR = Path('../03-output/results/comprehensive_16_stock')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    """Run comprehensive 16-stock analysis"""
    print("=" * 80)
    print("COMPREHENSIVE 16-STOCK DEVIATION ANALYSIS")
    print("=" * 80)
    print(f"\nTotal Stocks: {len(COMPREHENSIVE_STOCK_MAP)}")
    print(f"Sectors: {len(set([v['sector'] for v in COMPREHENSIVE_STOCK_MAP.values()]))}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Show breakdown by sector
    from collections import defaultdict
    by_sector = defaultdict(list)
    for stock, info in COMPREHENSIVE_STOCK_MAP.items():
        by_sector[info['sector']].append(stock)

    print("Stock Distribution:")
    for sector in sorted(by_sector.keys()):
        print(f"  {sector:25}: {', '.join(sorted(by_sector[sector]))}")
    print()

    # Run analysis
    analyzer = MultiSectorDeviationAnalysis(
        stock_sector_map=COMPREHENSIVE_STOCK_MAP,
        polarity_threshold=0.95
    )

    # Update output directory
    analyzer.output_dir = OUTPUT_DIR

    results_df = analyzer.run_multi_sector_analysis()

    # Additional summary statistics
    if results_df is not None and len(results_df) > 0:
        print("\n" + "=" * 80)
        print("SECTOR-LEVEL ANALYSIS")
        print("=" * 80)

        # Group by sector
        sector_stats = results_df.groupby('sector').agg({
            'deviation_increase_pct': ['mean', 'median', 'count'],
            'significant': 'sum',
            'news_days': 'mean'
        }).round(2)

        print("\nSector Summary:")
        print(sector_stats.to_string())

        # Save sector summary
        sector_stats.to_csv(OUTPUT_DIR / 'sector_level_summary.csv')

    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE 16-STOCK ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")

    return results_df


if __name__ == "__main__":
    import importlib.util
    main()
