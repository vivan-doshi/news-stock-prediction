"""
Run Phase 1 Analysis for Multiple Stocks
Analyzes Apple (AAPL) and Tesla (TSLA)
"""

from run_complete_pipeline import run_pipeline
from pathlib import Path

def analyze_multiple_stocks():
    """Run analysis for Apple and Tesla"""

    # Configuration
    stocks = [
        {
            'ticker': 'AAPL',
            'name': 'Apple Inc.',
            'sector_etf': 'XLK',  # Technology
            'start_date': '2020-01-01'
        },
        {
            'ticker': 'TSLA',
            'name': 'Tesla Inc.',
            'sector_etf': 'XLY',  # Consumer Discretionary (or could use XLE for Energy/Auto)
            'start_date': '2020-01-01'
        }
    ]

    print("\n" + "="*80)
    print(" "*25 + "MULTI-STOCK ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(stocks)} stocks: {', '.join([s['ticker'] for s in stocks])}")
    print("="*80 + "\n")

    results = {}

    for i, stock in enumerate(stocks, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(stocks)}] ANALYZING {stock['ticker']} - {stock['name']}")
        print(f"{'='*80}\n")

        # Create stock-specific output directory
        output_dir = f"../03-output/{stock['ticker']}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        try:
            # Run pipeline for this stock
            from run_complete_pipeline import CompletePipeline

            pipeline = CompletePipeline(
                ticker=stock['ticker'],
                start_date=stock['start_date'],
                end_date=None,  # Up to today
                sector_ticker=stock['sector_etf'],
                marketaux_api_key=None,  # Loads from .env
                data_dir="../01-data",
                output_dir=output_dir
            )

            pipeline.run_full_pipeline(skip_data_download=False)

            results[stock['ticker']] = {
                'status': 'SUCCESS',
                'summary': pipeline.summary
            }

        except Exception as e:
            print(f"\n❌ ERROR analyzing {stock['ticker']}: {e}")
            results[stock['ticker']] = {
                'status': 'FAILED',
                'error': str(e)
            }
            continue

    # Final summary
    print_final_summary(results)

    return results


def print_final_summary(results):
    """Print comparison summary for all stocks"""
    print("\n" + "="*80)
    print(" "*25 + "FINAL COMPARISON SUMMARY")
    print("="*80)

    for ticker, result in results.items():
        print(f"\n{ticker}:")
        print("-"*80)

        if result['status'] == 'SUCCESS':
            summary = result['summary']
            print(f"  ✅ Analysis Complete")
            print(f"  News Days: {summary['news_days']}")
            print(f"  Mean AR (News Days): {summary['mean_ar_news']*100:.2f}%")
            print(f"  Mean AR (Non-News Days): {summary['mean_ar_non_news']*100:.2f}%")
            print(f"  Model R²: {summary['avg_r_squared']:.3f}")
            print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")

            # Verdict
            if summary['significant_tests'] >= 3:
                print(f"  🎯 RESULT: News significantly impacts {ticker} stock price")
            elif summary['significant_tests'] >= 2:
                print(f"  ⚠️  RESULT: Partial evidence of news impact")
            else:
                print(f"  ❌ RESULT: Weak evidence of news impact")
        else:
            print(f"  ❌ Analysis Failed: {result['error']}")

    print("\n" + "="*80)
    print("All outputs saved to: 03-output/[TICKER]/")
    print("="*80)


if __name__ == "__main__":
    """
    Run analysis for Apple and Tesla

    Before running:
    1. Make sure you've installed dependencies: pip install -r requirements.txt
    2. Add your Finnhub API key to .env file
    3. Run: python run_analysis.py
    """

    print("\n🚀 Starting Multi-Stock Analysis Pipeline...\n")
    print("This will analyze:")
    print("  1. AAPL (Apple) - Technology sector")
    print("  2. TSLA (Tesla) - Consumer Discretionary sector")
    print("\nMake sure:")
    print("  ✓ Dependencies installed: pip install -r requirements.txt")
    print("  ✓ Marketaux API key added to .env file")

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    results = analyze_multiple_stocks()