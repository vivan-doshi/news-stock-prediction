"""
Final Event Study with Proper Filtering (26% Event Density)
Uses high-volume days only for clean event study
"""

import sys
import importlib
from pathlib import Path

# Import analysis module
main_analysis_module = importlib.import_module('05_main_analysis')
Phase1Analysis = main_analysis_module.Phase1Analysis

def run_final_event_study(ticker):
    """Run event study with high-volume filtering"""

    print(f"\n{'='*80}")
    print(f"EVENT STUDY: {ticker} (HIGH-VOLUME EVENTS ONLY)")
    print(f"{'='*80}")

    analysis = Phase1Analysis(
        stock_file=f'{ticker}_stock_data.csv',
        news_file=f'../03-output/filtered_analysis/{ticker}_event_dates_highvol.csv',
        ff_file='fama_french_factors.csv',
        sector_file=None,
        data_dir='../01-data',
        output_dir=f'../03-output/filtered_analysis/{ticker}_final_study'
    )

    summary = analysis.run_complete_analysis()

    return summary

def main():
    """Run final event studies for both tickers"""

    print("\n" + "#"*80)
    print("#"*80)
    print("###" + " "*74 + "###")
    print("###" + "   FINAL EVENT STUDY ANALYSIS".center(74) + "###")
    print("###" + "   High-Volume Events Only (26% Density)".center(74) + "###")
    print("###" + " "*74 + "###")
    print("#"*80)
    print("#"*80)

    results = {}

    for ticker in ['TSLA', 'AAPL']:
        try:
            summary = run_final_event_study(ticker)
            results[ticker] = summary
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = None

    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    for ticker, summary in results.items():
        if summary:
            print(f"\n{ticker}:")
            print(f"  Event Days: {summary['news_days']:,}")
            print(f"  Non-Event Days: {summary['non_news_days']:,}")
            print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.3f}%")
            print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.3f}%")
            print(f"  Difference: {(summary['mean_ar_news'] - summary['mean_ar_non_news'])*100:.3f}%")
            print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
            print(f"  Model R²: {summary['avg_r_squared']:.3f}")

            if summary['significant_tests'] >= 2:
                print(f"  ✅ Strong evidence of news impact")
            elif summary['significant_tests'] >= 1:
                print(f"  ⚠️  Moderate evidence of news impact")
            else:
                print(f"  ❓ Limited statistical significance")

    print("\n" + "="*80)
    print("OUTPUT LOCATIONS:")
    print("="*80)
    print("  • ../03-output/filtered_analysis/TSLA_final_study/")
    print("  • ../03-output/filtered_analysis/AAPL_final_study/")
    print("\nFiles include:")
    print("  - abnormal_returns.csv")
    print("  - beta_estimates.csv")
    print("  - statistical_tests.csv")
    print("  - Visualizations (PNG files)")
    print("="*80)

if __name__ == "__main__":
    main()
