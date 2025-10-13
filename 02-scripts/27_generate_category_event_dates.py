"""
GENERATE CATEGORY-SPECIFIC EVENT DATES
=======================================

Extracts event dates for each news category from balanced filtered news.

Input: AAPL_balanced_filtered.csv (has category columns)
Output:
- category_event_study/event_dates/AAPL/Earnings_events.csv
- category_event_study/event_dates/AAPL/Product_Launch_events.csv
- etc.

Categories:
- Earnings
- Product Launch
- Executive Changes
- M&A
- Regulatory/Legal
- Analyst Ratings
- Dividends
- Market Performance

Author: Category Event Study System
Date: 2025-10-13
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import stock configuration
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import importlib
config_module = importlib.import_module('21_expanded_50_stock_config')
STOCKS = config_module.EXPANDED_STOCKS

# Parameters
SCRIPT_DIR = Path(__file__).parent
NEWS_DIR = SCRIPT_DIR.parent / "03-output" / "news_filtering_comparison"
OUTPUT_DIR = SCRIPT_DIR.parent / "03-output" / "category_event_study" / "event_dates"

# News categories (must match column names in filtered CSVs)
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


def extract_category_events(ticker: str) -> Dict[str, pd.DataFrame]:
    """
    Extract event dates for each category for a given stock

    Returns:
        Dictionary mapping category name to DataFrame of events
    """
    print(f"\n{'='*60}")
    print(f"Processing {ticker}")
    print(f"{'='*60}")

    # Load balanced filtered news
    news_file = NEWS_DIR / f"{ticker}_balanced_filtered.csv"

    if not news_file.exists():
        print(f"⚠️  No news file found: {news_file}")
        return {}

    try:
        df = pd.read_csv(news_file)
        df['date'] = pd.to_datetime(df['date'])

        print(f"Total news articles: {len(df)}")

        category_events = {}

        for category in CATEGORIES:
            cat_col = f"cat_{category}"

            if cat_col not in df.columns:
                print(f"⚠️  Category column not found: {cat_col}")
                continue

            # Filter news for this category
            category_df = df[df[cat_col] == True].copy()

            if len(category_df) == 0:
                print(f"  {category}: 0 events (skipping)")
                continue

            # Extract unique dates (one event per day)
            category_df['event_date'] = category_df['date'].dt.date
            unique_dates = category_df.drop_duplicates(subset=['event_date'])

            # Create event dataframe
            events = unique_dates[['event_date', 'title', 'sentiment_polarity', 'primary_category']].copy()
            events = events.sort_values('event_date')

            category_events[category] = events

            print(f"  {category}: {len(events)} unique event dates")

        return category_events

    except Exception as e:
        print(f"❌ Error processing {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


def save_category_events(ticker: str, category_events: Dict[str, pd.DataFrame]):
    """Save category event dates to CSV files"""
    ticker_dir = OUTPUT_DIR / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    for category, events_df in category_events.items():
        # Clean category name for filename
        clean_category = category.replace('/', '_').replace(' ', '_')
        output_file = ticker_dir / f"{clean_category}_events.csv"

        events_df.to_csv(output_file, index=False)
        print(f"  ✅ Saved: {output_file.name}")


def generate_summary_statistics(all_results: Dict[str, Dict[str, pd.DataFrame]]):
    """Generate summary statistics across all stocks and categories"""
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY STATISTICS")
    print(f"{'='*80}")

    # Collect statistics
    summary_data = []

    for ticker, category_events in all_results.items():
        if ticker not in STOCKS:
            continue

        sector = STOCKS[ticker]['sector']

        for category, events_df in category_events.items():
            summary_data.append({
                'ticker': ticker,
                'sector': sector,
                'category': category,
                'num_events': len(events_df),
                'avg_sentiment': events_df['sentiment_polarity'].mean() if len(events_df) > 0 else np.nan,
                'positive_sentiment_pct': (events_df['sentiment_polarity'] > 0).sum() / len(events_df) * 100 if len(events_df) > 0 else 0,
                'negative_sentiment_pct': (events_df['sentiment_polarity'] < 0).sum() / len(events_df) * 100 if len(events_df) > 0 else 0
            })

    summary_df = pd.DataFrame(summary_data)

    # Save overall summary
    summary_file = OUTPUT_DIR.parent / "category_summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n✅ Saved: {summary_file}")

    # Print summary by category
    print("\n" + "="*80)
    print("SUMMARY BY CATEGORY")
    print("="*80)

    category_summary = summary_df.groupby('category').agg({
        'num_events': ['sum', 'mean', 'std'],
        'avg_sentiment': 'mean',
        'ticker': 'count'
    }).round(2)

    category_summary.columns = ['Total_Events', 'Avg_Events_Per_Stock', 'Std_Events', 'Avg_Sentiment', 'Num_Stocks']
    print(category_summary.to_string())

    # Print summary by sector
    print("\n" + "="*80)
    print("SUMMARY BY SECTOR")
    print("="*80)

    sector_summary = summary_df.groupby('sector').agg({
        'num_events': ['sum', 'mean'],
        'ticker': lambda x: len(x.unique())
    }).round(2)

    sector_summary.columns = ['Total_Events', 'Avg_Events_Per_Stock', 'Num_Stocks']
    print(sector_summary.to_string())

    return summary_df


def create_category_sector_matrix(summary_df: pd.DataFrame):
    """Create category × sector matrix visualization data"""
    print(f"\n{'='*80}")
    print("CREATING CATEGORY × SECTOR MATRIX")
    print(f"{'='*80}")

    # Pivot table: sectors as rows, categories as columns
    matrix = summary_df.pivot_table(
        index='sector',
        columns='category',
        values='num_events',
        aggfunc='sum',
        fill_value=0
    )

    # Save matrix
    matrix_file = OUTPUT_DIR.parent / "category_sector_matrix.csv"
    matrix.to_csv(matrix_file)
    print(f"\n✅ Saved: {matrix_file}")

    print("\nCategory × Sector Event Count Matrix:")
    print(matrix.to_string())

    return matrix


def main():
    """Generate category-specific event dates for all stocks"""
    print("="*80)
    print("CATEGORY EVENT DATE EXTRACTION")
    print("="*80)
    print(f"\nStocks: {len(STOCKS)}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"  {', '.join(CATEGORIES)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Process each stock
    for i, ticker in enumerate(STOCKS.keys(), 1):
        print(f"\n[{i}/{len(STOCKS)}] Processing {ticker}...")

        category_events = extract_category_events(ticker)

        if category_events:
            save_category_events(ticker, category_events)
            all_results[ticker] = category_events
        else:
            print(f"  ⚠️  No category events extracted for {ticker}")

    # Generate summary statistics
    if all_results:
        summary_df = generate_summary_statistics(all_results)
        matrix = create_category_sector_matrix(summary_df)

    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Successfully processed {len(all_results)} stocks")
    print(f"📁 Event dates saved to: {OUTPUT_DIR}")
    print(f"📊 Summary statistics saved to: {OUTPUT_DIR.parent}")

    return all_results


if __name__ == "__main__":
    results = main()