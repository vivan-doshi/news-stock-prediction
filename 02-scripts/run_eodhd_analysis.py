"""
Complete Analysis Pipeline Using EODHD Data
Runs the full event study analysis for both AAPL and TSLA
"""

import sys
import importlib
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
main_analysis_module = importlib.import_module('05_main_analysis')
Phase1Analysis = main_analysis_module.Phase1Analysis


def prepare_news_dates(news_file: str, output_file: str, data_dir: str = "../01-data"):
    """
    Prepare news dates file from EODHD news CSV

    Parameters:
    -----------
    news_file : str
        Input news file (e.g., 'AAPL_eodhd_news.csv')
    output_file : str
        Output file name (e.g., 'AAPL_news_dates.csv')
    """
    news_path = Path(data_dir) / news_file

    if not news_path.exists():
        print(f"  ⚠ News file not found: {news_path}")
        return False

    # Read EODHD news
    df = pd.read_csv(news_path)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Get unique dates
    news_dates = pd.DataFrame({'Date': df['date'].dt.date.unique()})
    news_dates = news_dates.sort_values('Date')

    # Save
    output_path = Path(data_dir) / output_file
    news_dates.to_csv(output_path, index=False)

    print(f"  ✓ Created {output_file} with {len(news_dates)} unique news dates")
    return True


def run_analysis_for_ticker(ticker: str, start_date: str, end_date: str):
    """
    Run complete analysis for a single ticker

    Parameters:
    -----------
    ticker : str
        Stock ticker (e.g., 'AAPL', 'TSLA')
    start_date : str
        Start date
    end_date : str
        End date
    """
    print("\n" + "=" * 80)
    print(f"RUNNING ANALYSIS FOR {ticker}")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")

    # File names
    stock_file = f"{ticker}_stock_data.csv"
    news_file = f"{ticker}_eodhd_news.csv"
    news_dates_file = f"{ticker}_news_dates.csv"
    ff_file = "fama_french_factors.csv"  # Using pre-processed Fama-French file
    output_dir = f"../03-output/{ticker}"

    # Step 1: Prepare news dates
    print("\n[Prep] Preparing news dates file...")
    success = prepare_news_dates(news_file, news_dates_file)

    if not success:
        print(f"\n❌ Failed to prepare news dates for {ticker}")
        return None

    # Step 2: Run Phase 1 analysis
    print("\n[Analysis] Starting Phase 1 event study...")

    try:
        analysis = Phase1Analysis(
            stock_file=stock_file,
            news_file=news_dates_file,
            ff_file=ff_file,
            sector_file=None,  # Not using sector factors
            data_dir="../01-data",
            output_dir=output_dir
        )

        summary = analysis.run_complete_analysis()

        print(f"\n✅ {ticker} Analysis Complete!")
        print(f"   Output saved to: {Path(output_dir).absolute()}")

        return summary

    except Exception as e:
        print(f"\n❌ {ticker} Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run complete analysis for both AAPL and TSLA"""

    print("\n" + "=" * 80)
    print(" " * 20 + "COMPLETE EODHD NEWS-STOCK ANALYSIS")
    print("=" * 80)
    print("\nThis will run Phase 1 event study analysis for:")
    print("  - AAPL (Apple Inc.)")
    print("  - TSLA (Tesla Inc.)")
    print("\nUsing EODHD news data and Fama-French 5-factor model")
    print("=" * 80)

    results = {}

    # Analyze AAPL
    print("\n\n" + "🍎" * 40)
    results['AAPL'] = run_analysis_for_ticker(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-31'
    )

    # Analyze TSLA
    print("\n\n" + "⚡" * 40)
    results['TSLA'] = run_analysis_for_ticker(
        ticker='TSLA',
        start_date='2020-01-01',
        end_date='2025-10-08'
    )

    # Final Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL SUMMARY")
    print("=" * 80)

    for ticker, summary in results.items():
        if summary:
            print(f"\n{ticker}:")
            print(f"  News Days: {summary['news_days']}")
            print(f"  Mean AR (News): {summary['mean_ar_news']*100:.2f}%")
            print(f"  Mean AR (Non-News): {summary['mean_ar_non_news']*100:.2f}%")
            print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
            print(f"  Model R²: {summary['avg_r_squared']:.3f}")

            # Success evaluation
            if summary['significant_tests'] >= 3:
                print(f"  ✅ Strong evidence of news impact!")
            elif summary['significant_tests'] >= 2:
                print(f"  ⚠️  Some evidence of news impact")
            else:
                print(f"  ❌ Weak evidence of news impact")
        else:
            print(f"\n{ticker}: ❌ Analysis failed")

    print("\n" + "=" * 80)
    print("🎉 ALL ANALYSES COMPLETE!")
    print("=" * 80)
    print("\n📁 Results saved to:")
    print("   - 03-output/AAPL/")
    print("   - 03-output/TSLA/")
    print("\nKey output files:")
    print("   - abnormal_returns.csv")
    print("   - beta_estimates.csv")
    print("   - statistical_tests.csv")
    print("   - analysis_summary.png")
    print("   - analysis_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
