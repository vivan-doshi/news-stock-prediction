"""
Create stricter filtering for event study
Focus on truly significant events only
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_strict_event_dates(ticker):
    """Create event dates with strict filtering - one event per day max"""

    data_dir = Path("../01-data")
    filtered_dir = Path("../03-output/filtered_analysis")

    print(f"\n{'='*80}")
    print(f"CREATING STRICT EVENT DATES FOR {ticker}")
    print(f"{'='*80}")

    # Load the categorized news
    news_file = filtered_dir / f"{ticker}_news_categorized.csv"
    df = pd.read_csv(news_file)
    df['date'] = pd.to_datetime(df['date'])

    # Load stock data to align dates
    stock_file = data_dir / f"{ticker}_stock_data.csv"
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)

    print(f"\n📊 Data Overview:")
    print(f"  Total articles: {len(df):,}")
    print(f"  Trading days: {len(stock_df):,}")

    # Apply MUCH stricter filtering
    print(f"\n🔍 Applying Strict Filters:")

    # Filter 1: Only high-priority event categories
    priority_categories = ['earnings', 'product', 'executive']
    priority_filter = df['primary_category'].isin(priority_categories)
    print(f"  1. Priority categories only: {priority_filter.sum():,} articles")

    # Filter 2: Strong sentiment (stricter threshold)
    sentiment_filter = df['sentiment_polarity'].abs() > 0.7
    print(f"  2. Extreme sentiment (|pol| > 0.7): {sentiment_filter.sum():,} articles")

    # Filter 3: High-volume days (top 10%)
    daily_counts = df.groupby(df['date'].dt.date).size()
    threshold_90 = daily_counts.quantile(0.90)
    df['daily_count'] = df['date'].dt.date.map(daily_counts.to_dict())
    volume_filter = df['daily_count'] >= threshold_90
    print(f"  3. High-volume days (≥{threshold_90:.0f} articles): {volume_filter.sum():,} articles")

    # Combined filter: Must meet at least 2 of the 3 criteria
    df['filter_score'] = priority_filter.astype(int) + sentiment_filter.astype(int) + volume_filter.astype(int)
    strict_filter = df['filter_score'] >= 2

    print(f"\n  Combined (≥2 criteria): {strict_filter.sum():,} articles")

    filtered_df = df[strict_filter].copy()

    # Group by date and select ONE article per day (highest sentiment magnitude)
    print(f"\n📅 Selecting one event per day...")
    filtered_df['sentiment_abs'] = filtered_df['sentiment_polarity'].abs()

    # For each date, keep the article with:
    # 1. Priority category first
    # 2. Highest sentiment magnitude
    filtered_df['priority_score'] = filtered_df['primary_category'].map({
        'earnings': 3,
        'product': 2,
        'executive': 1
    }).fillna(0)

    # Keep one article per day (highest priority, then highest sentiment)
    event_dates_df = (filtered_df
                      .sort_values(['date', 'priority_score', 'sentiment_abs'], ascending=[True, False, False])
                      .groupby(filtered_df['date'].dt.date)
                      .first()
                      .reset_index(drop=True))

    print(f"  Unique event dates: {len(event_dates_df):,}")

    # Align with trading days
    event_dates_df['Date'] = pd.to_datetime(event_dates_df['date']).dt.tz_localize(None)
    stock_min = stock_df['Date'].min()
    stock_max = stock_df['Date'].max()
    event_dates_df = event_dates_df[
        (event_dates_df['Date'] >= stock_min) &
        (event_dates_df['Date'] <= stock_max)
    ]

    print(f"  After aligning with trading days: {len(event_dates_df):,}")

    # Calculate event density
    event_density = len(event_dates_df) / len(stock_df) * 100
    print(f"\n📈 Event Density: {event_density:.1f}%")

    if event_density > 50:
        print(f"  ⚠️  Still high - consider reducing further")
    elif event_density > 30:
        print(f"  ✅ Good density for event study")
    else:
        print(f"  ✅ Excellent - focused on truly significant events")

    # Show category breakdown
    print(f"\n📋 Event Categories:")
    category_counts = event_dates_df['primary_category'].value_counts()
    for cat, count in category_counts.items():
        print(f"  {cat}: {count:,} ({count/len(event_dates_df)*100:.1f}%)")

    # Save event dates
    output_df = event_dates_df[['Date']].copy()
    output_file = filtered_dir / f"{ticker}_event_dates_strict.csv"
    output_df.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file.name}")

    # Also save the full filtered dataset for analysis
    detailed_file = filtered_dir / f"{ticker}_events_strict_detailed.csv"
    event_dates_df.to_csv(detailed_file, index=False)
    print(f"💾 Saved detailed info to: {detailed_file.name}")

    return event_dates_df


def create_alternative_strategies(ticker):
    """Create alternative filtering strategies for comparison"""

    filtered_dir = Path("../03-output/filtered_analysis")
    data_dir = Path("../01-data")

    print(f"\n{'='*80}")
    print(f"ALTERNATIVE STRATEGIES FOR {ticker}")
    print(f"{'='*80}")

    # Load data
    news_file = filtered_dir / f"{ticker}_news_categorized.csv"
    df = pd.read_csv(news_file)
    df['date'] = pd.to_datetime(df['date'])

    stock_file = data_dir / f"{ticker}_stock_data.csv"
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)

    strategies = {}

    # Strategy A: Earnings only
    earnings_df = df[df['category_earnings']].copy()
    earnings_dates = earnings_df.groupby(earnings_df['date'].dt.date).first().reset_index(drop=True)
    earnings_dates['Date'] = pd.to_datetime(earnings_dates['date']).dt.tz_localize(None)
    stock_min = stock_df['Date'].min()
    stock_max = stock_df['Date'].max()
    earnings_dates = earnings_dates[
        (earnings_dates['Date'] >= stock_min) &
        (earnings_dates['Date'] <= stock_max)
    ]
    strategies['earnings_only'] = len(earnings_dates)

    # Strategy B: Top 20% volume days only
    daily_counts = df.groupby(df['date'].dt.date).size()
    threshold_80 = daily_counts.quantile(0.80)
    high_vol_dates = daily_counts[daily_counts >= threshold_80].index
    strategies['high_volume'] = len(high_vol_dates)

    # Strategy C: Extreme sentiment only
    extreme_df = df[df['sentiment_polarity'].abs() > 0.8].copy()
    extreme_dates = extreme_df.groupby(extreme_df['date'].dt.date).first().reset_index(drop=True)
    extreme_dates['Date'] = pd.to_datetime(extreme_dates['date']).dt.tz_localize(None)
    stock_min = stock_df['Date'].min()
    stock_max = stock_df['Date'].max()
    extreme_dates = extreme_dates[
        (extreme_dates['Date'] >= stock_min) &
        (extreme_dates['Date'] <= stock_max)
    ]
    strategies['extreme_sentiment'] = len(extreme_dates)

    print(f"\n📊 Strategy Comparison:")
    print(f"  Trading days: {len(stock_df):,}")
    for name, count in strategies.items():
        density = count / len(stock_df) * 100
        print(f"  {name}: {count:,} events ({density:.1f}% density)")

    return strategies


def main():
    """Create strict event dates for both tickers"""
    print("\n" + "="*80)
    print("STRICT EVENT FILTERING TOOL")
    print("="*80)
    print("\nThis creates event dates suitable for traditional event studies")
    print("Target: 20-40% event density (1 event per 2.5-5 trading days)")

    results = {}

    for ticker in ['TSLA', 'AAPL']:
        event_df = create_strict_event_dates(ticker)
        strategies = create_alternative_strategies(ticker)
        results[ticker] = {
            'events': len(event_df),
            'strategies': strategies
        }

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for ticker, data in results.items():
        print(f"\n{ticker}:")
        print(f"  Strict filter: {data['events']:,} events")
        print(f"  Alternative strategies:")
        for name, count in data['strategies'].items():
            print(f"    - {name}: {count:,}")

    print("\n" + "="*80)
    print("✅ STRICT FILTERING COMPLETE")
    print("="*80)
    print("\nFiles created:")
    print("  • TSLA_event_dates_strict.csv")
    print("  • AAPL_event_dates_strict.csv")
    print("  • TSLA_events_strict_detailed.csv")
    print("  • AAPL_events_strict_detailed.csv")
    print("\nNext: Run event study with *_event_dates_strict.csv files")
    print("="*80)


if __name__ == "__main__":
    main()
