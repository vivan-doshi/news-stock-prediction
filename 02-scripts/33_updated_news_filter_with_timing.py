"""
UPDATED NEWS FILTERING WITH 3:30 PM ET CUTOFF
===============================================

Applies comprehensive news filtering with:
1. 3:30 PM ET timing adjustment (already done in raw data)
2. Event categorization
3. Balanced filter criteria
4. False positive detection

This script processes the UPDATED news data (2021-2025) and creates
filtered event files for event study analysis.

Author: Updated Analysis Pipeline
Date: 2025-10-13
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import stock configuration
import sys
sys.path.insert(0, str(Path(__file__).parent))
import importlib
config_module = importlib.import_module('21_expanded_50_stock_config')
STOCKS = config_module.EXPANDED_STOCKS

# Parameters
DATA_DIR = Path(__file__).parent.parent / "01-data"
OUTPUT_DIR = Path(__file__).parent.parent / "03-output" / "news_filtering_2021_2025"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Event categories
EVENT_CATEGORIES = {
    'Earnings': ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4',
                'revenue', 'profit', 'eps', 'beats estimates', 'misses estimates', 'guidance'],
    'Product Launch': ['launch', 'release', 'unveil', 'introduce', 'new product', 'new model',
                     'announces', 'unveiled', 'releasing', 'debut'],
    'Executive Changes': ['ceo', 'executive', 'leadership', 'management', 'board', 'resign',
                        'appoint', 'hire', 'chief', 'president', 'director', 'chairman'],
    'M&A': ['merger', 'acquisition', 'acquire', 'acquiring', 'buyout', 'takeover',
           'joint venture', 'strategic partnership', 'deal', 'combine'],
    'Regulatory/Legal': ['sec', 'lawsuit', 'investigation', 'probe', 'regulator', 'compliance',
                       'legal', 'fine', 'settlement', 'court', 'judge', 'ftc', 'doj'],
    'Analyst Ratings': ['upgrade', 'downgrade', 'rating', 'target price', 'analyst',
                      'price target', 'outperform', 'underperform', 'buy', 'sell', 'initiate'],
    'Dividends': ['dividend', 'payout', 'distribution', 'yield', 'dividend increase', 'special dividend'],
    'Market Performance': ['stock', 'shares', 'market', 'trading', 'price', 'rally',
                         'drop', 'surge', 'plunge', 'investors', 'wall street']
}

def categorize_news(df, ticker):
    """
    Apply event categorization to news data

    Returns DataFrame with category columns (cat_Earnings, cat_Product Launch, etc.)
    """
    print(f"  → Categorizing events...")

    for category, keywords in EVENT_CATEGORIES.items():
        pattern = '|'.join([f'\\b{kw}\\b' for kw in keywords])

        # Check in title (higher weight) and content
        title_match = df['title'].str.contains(pattern, case=False, na=False, regex=True)
        content_match = df['content'].str.contains(pattern, case=False, na=False, regex=True)

        df[f'cat_{category}'] = title_match | content_match

    # Count categories per article
    cat_cols = [f'cat_{cat}' for cat in EVENT_CATEGORIES.keys()]
    df['num_categories'] = df[cat_cols].sum(axis=1)

    # Assign primary category (first match in priority order)
    priority_order = ['Earnings', 'Product Launch', 'Regulatory/Legal', 'Analyst Ratings',
                     'Executive Changes', 'Dividends', 'M&A', 'Market Performance']

    def get_primary_category(row):
        for cat in priority_order:
            if row[f'cat_{cat}']:
                return cat
        return 'Uncategorized'

    df['primary_category'] = df.apply(get_primary_category, axis=1)

    # Print category distribution
    cat_counts = df['primary_category'].value_counts()
    print(f"    Category distribution:")
    for cat, count in cat_counts.head(8).items():
        print(f"      {cat}: {count} ({count/len(df)*100:.1f}%)")

    return df

def detect_false_positives(df, ticker, company_name):
    """
    Detect false positive indicators
    """
    print(f"  → Detecting false positives...")

    # 1. Ticker in title
    df['ticker_in_title'] = df['title'].str.contains(
        ticker, case=False, na=False, regex=False
    )

    # Also check for company name in title
    company_keywords = company_name.split()[:2]  # First 2 words
    for keyword in company_keywords:
        if len(keyword) > 3:  # Skip short words
            df['ticker_in_title'] |= df['title'].str.contains(
                keyword, case=False, na=False, regex=False
            )

    # 2. Count tickers mentioned
    def count_tickers(symbols_str):
        if pd.isna(symbols_str):
            return 1
        tickers = [s.strip() for s in str(symbols_str).split(',')]
        return len([t for t in tickers if len(t) > 0])

    df['ticker_count'] = df['symbols'].apply(count_tickers)

    # 3. Content quality
    df['content_length'] = df['content'].astype(str).str.len()
    df['title_length'] = df['title'].astype(str).str.len()

    # 4. False positive score (0 = best, 3 = worst)
    df['fp_score'] = 0
    df.loc[~df['ticker_in_title'], 'fp_score'] += 1
    df.loc[df['ticker_count'] > 2, 'fp_score'] += 1
    df.loc[df['content_length'] < 200, 'fp_score'] += 1

    print(f"    • Ticker in title: {df['ticker_in_title'].sum()} ({df['ticker_in_title'].mean()*100:.1f}%)")
    print(f"    • Single ticker: {(df['ticker_count'] == 1).sum()} ({(df['ticker_count'] == 1).mean()*100:.1f}%)")
    print(f"    • FP Score 0 (clean): {(df['fp_score'] == 0).sum()} ({(df['fp_score'] == 0).mean()*100:.1f}%)")

    return df

def apply_balanced_filter(df, ticker):
    """
    Apply balanced filter criteria:
    - Strong sentiment (|polarity| > 0.5) OR ticker in title
    - Matches at least one priority category
    - Content length >= 100 chars
    - Max 3 tickers mentioned
    - One event per day (select strongest)
    """
    print(f"  → Applying balanced filter...")

    # Filter criteria
    strong_sentiment = df['sentiment_polarity'].abs() > 0.5
    ticker_in_title = df['ticker_in_title'] == True
    priority_categories = df['primary_category'].isin([
        'Earnings', 'Product Launch', 'Regulatory/Legal', 'Analyst Ratings',
        'Executive Changes', 'Dividends', 'M&A'
    ])
    good_content = df['content_length'] >= 100
    few_tickers = df['ticker_count'] <= 3

    # Combined filter
    balanced_mask = (
        (strong_sentiment | ticker_in_title) &
        priority_categories &
        good_content &
        few_tickers
    )

    df_filtered = df[balanced_mask].copy()

    print(f"    • Before filter: {len(df)} articles")
    print(f"    • After filter: {len(df_filtered)} articles ({len(df_filtered)/len(df)*100:.1f}%)")

    # Select one event per day (highest priority + strongest sentiment)
    if len(df_filtered) > 0:
        # Sort by event_date, priority, and sentiment strength
        priority_map = {cat: i for i, cat in enumerate([
            'Earnings', 'Product Launch', 'Regulatory/Legal', 'Analyst Ratings',
            'Executive Changes', 'Dividends', 'M&A'
        ])}

        df_filtered['priority_rank'] = df_filtered['primary_category'].map(priority_map)
        df_filtered['sentiment_strength'] = df_filtered['sentiment_polarity'].abs()

        # Keep one per event_date
        df_filtered = df_filtered.sort_values(
            ['event_date', 'priority_rank', 'sentiment_strength'],
            ascending=[True, True, False]
        ).groupby('event_date').first().reset_index()

        print(f"    • One per day: {len(df_filtered)} events")

    return df_filtered

def process_stock(ticker):
    """Process news data for a single stock"""
    company_name = STOCKS[ticker]['name']
    sector = STOCKS[ticker]['sector']

    print(f"\n{'='*60}")
    print(f"{ticker} - {company_name} ({sector})")
    print(f"{'='*60}")

    # Load news data
    news_file = DATA_DIR / f"{ticker}_eodhd_news.csv"
    if not news_file.exists():
        print(f"  ⚠️  No news file found")
        return None

    df = pd.read_csv(news_file)

    # Convert dates
    df['date'] = pd.to_datetime(df['date'])

    # Use event_date if exists, otherwise use date
    if 'event_date' not in df.columns:
        df['event_date'] = df['date']
    else:
        df['event_date'] = pd.to_datetime(df['event_date'])

    print(f"  ✓ Loaded: {len(df)} articles")
    print(f"    Date range: {df['date'].min()} to {df['date'].max()}")

    # Filter to 2021-2025 analysis window
    analysis_start = pd.Timestamp('2021-01-01')
    analysis_end = pd.Timestamp('2025-07-01')

    df = df[(df['event_date'] >= analysis_start) & (df['event_date'] <= analysis_end)].copy()
    print(f"    Analysis window (2021-2025): {len(df)} articles")

    if len(df) == 0:
        print(f"  ⚠️  No articles in analysis window")
        return None

    # Apply processing steps
    df = categorize_news(df, ticker)
    df = detect_false_positives(df, ticker, company_name)
    df_filtered = apply_balanced_filter(df, ticker)

    if len(df_filtered) == 0:
        print(f"  ⚠️  No articles passed filter")
        return None

    # Save filtered data
    output_file = OUTPUT_DIR / f"{ticker}_balanced_filtered.csv"
    df_filtered.to_csv(output_file, index=False)
    print(f"  ✅ Saved: {output_file}")

    # Return summary stats
    return {
        'ticker': ticker,
        'total_articles': len(df),
        'filtered_events': len(df_filtered),
        'date_range_start': df['event_date'].min(),
        'date_range_end': df['event_date'].max(),
        'primary_categories': df_filtered['primary_category'].value_counts().to_dict()
    }

def main():
    print("="*80)
    print("UPDATED NEWS FILTERING WITH 3:30 PM ET CUTOFF")
    print("="*80)
    print(f"Analysis window: 2021-01-01 to 2025-07-01")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Stocks: {len(STOCKS)}")

    results = []

    for i, ticker in enumerate(sorted(STOCKS.keys()), 1):
        print(f"\n[{i}/{len(STOCKS)}]", end=" ")
        result = process_stock(ticker)
        if result:
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("FILTERING SUMMARY")
    print("="*80)

    if results:
        df_summary = pd.DataFrame(results)

        print(f"\n✅ Successfully processed: {len(results)}/{len(STOCKS)} stocks")
        print(f"\nTotal articles: {df_summary['total_articles'].sum():,}")
        print(f"Total filtered events: {df_summary['filtered_events'].sum():,}")
        print(f"Average events per stock: {df_summary['filtered_events'].mean():.1f}")
        print(f"Median events per stock: {df_summary['filtered_events'].median():.0f}")

        # Save summary
        summary_file = OUTPUT_DIR / "filtering_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"\n💾 Saved summary: {summary_file}")

        # Top stocks by events
        print(f"\nTop 10 stocks by filtered events:")
        top10 = df_summary.nlargest(10, 'filtered_events')
        for _, row in top10.iterrows():
            print(f"  {row['ticker']}: {row['filtered_events']} events")

    print(f"\n✅ Complete! Filtered data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()