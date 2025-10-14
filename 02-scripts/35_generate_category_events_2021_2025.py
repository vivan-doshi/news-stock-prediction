"""
GENERATE CATEGORY EVENT DATES (2021-2025)
==========================================

Creates event date files for each category × stock combination
from the filtered news data.

Input: news_filtering_2021_2025/[TICKER]_balanced_filtered.csv
Output: 03-output/event_study_2021_2025/event_dates/[TICKER]/[CATEGORY]_events.csv

Author: Category Event Generator
Date: 2025-10-13
"""

import pandas as pd
from pathlib import Path
import sys

# Import stock configuration
sys.path.insert(0, str(Path(__file__).parent))
import importlib
config_module = importlib.import_module('21_expanded_50_stock_config')
STOCKS = config_module.EXPANDED_STOCKS

# Parameters
SCRIPT_DIR = Path(__file__).parent
NEWS_DIR = SCRIPT_DIR.parent / "03-output" / "news_filtering_2021_2025"
OUTPUT_DIR = SCRIPT_DIR.parent / "03-output" / "event_study_2021_2025" / "event_dates"

# Categories
CATEGORIES = [
    'Earnings',
    'Product Launch',
    'Executive Changes',
    'M&A',
    'Regulatory/Legal',
    'Analyst Ratings',
    'Dividends',
    'Market Performance'
]

def generate_events_for_stock(ticker):
    """Generate category-specific event files for one stock"""
    print(f"\n{'='*60}")
    print(f"{ticker} - {STOCKS[ticker]['name']}")
    print(f"{'='*60}")

    # Load filtered news
    news_file = NEWS_DIR / f"{ticker}_balanced_filtered.csv"

    if not news_file.exists():
        print(f"  ⚠️  No filtered news file found")
        return None

    df = pd.read_csv(news_file)
    df['event_date'] = pd.to_datetime(df['event_date'])

    print(f"  ✓ Loaded: {len(df)} filtered events")

    # Create output directory for this ticker
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(exist_ok=True, parents=True)

    category_counts = {}

    # Extract events for each category
    for category in CATEGORIES:
        cat_col = f"cat_{category}"

        if cat_col not in df.columns:
            print(f"  ⚠️  Category column not found: {cat_col}")
            continue

        # Filter for this category
        category_df = df[df[cat_col] == True].copy()

        if len(category_df) == 0:
            print(f"  {category}: 0 events (skipping)")
            continue

        # Get unique event dates
        event_dates = category_df[['event_date']].drop_duplicates().sort_values('event_date')

        # Save
        output_file = ticker_dir / f"{category.replace(' ', '_')}_events.csv"
        event_dates.to_csv(output_file, index=False)

        category_counts[category] = len(event_dates)
        print(f"  ✓ {category}: {len(event_dates)} events → {output_file.name}")

    # Save summary
    summary_df = pd.DataFrame([
        {'category': cat, 'event_count': count}
        for cat, count in category_counts.items()
    ])
    summary_file = ticker_dir / "_category_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    return {
        'ticker': ticker,
        'total_events': len(df),
        'categories': category_counts
    }

def main():
    print("="*80)
    print("GENERATE CATEGORY EVENT DATES (2021-2025)")
    print("="*80)
    print(f"Input: {NEWS_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Stocks: {len(STOCKS)}")
    print(f"Categories: {len(CATEGORIES)}")

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    results = []

    for i, ticker in enumerate(sorted(STOCKS.keys()), 1):
        print(f"\n[{i}/{len(STOCKS)}]", end=" ")
        result = generate_events_for_stock(ticker)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80)

    if results:
        print(f"\n✅ Successfully processed: {len(results)}/{len(STOCKS)} stocks")

        # Category statistics across all stocks
        all_categories = {}
        for result in results:
            for cat, count in result['categories'].items():
                if cat not in all_categories:
                    all_categories[cat] = []
                all_categories[cat].append(count)

        print(f"\nCategory statistics across all stocks:")
        print(f"{'Category':<25} {'Total Events':<15} {'Avg/Stock':<15} {'Stocks with Events'}")
        print("-" * 80)

        for category in CATEGORIES:
            if category in all_categories:
                counts = all_categories[category]
                total = sum(counts)
                avg = total / len(results)
                num_stocks = len(counts)

                print(f"{category:<25} {total:<15,} {avg:<15.1f} {num_stocks}/{len(STOCKS)}")

        # Save overall summary
        summary_file = OUTPUT_DIR / "overall_summary.csv"
        summary_data = []
        for result in results:
            for cat, count in result['categories'].items():
                summary_data.append({
                    'ticker': result['ticker'],
                    'category': cat,
                    'event_count': count
                })

        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        print(f"\n💾 Saved overall summary: {summary_file}")

    print(f"\n✅ Complete! Event dates saved to: {OUTPUT_DIR}")
    print(f"\n📂 Total files created: {len(list(OUTPUT_DIR.rglob('*.csv')))}")

if __name__ == "__main__":
    main()