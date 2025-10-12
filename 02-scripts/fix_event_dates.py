"""
Fix event dates to align with stock data availability
"""

import pandas as pd
from pathlib import Path

def fix_event_dates(ticker):
    """Fix event dates to match stock data date range"""

    data_dir = Path("../01-data")
    filtered_dir = Path("../03-output/filtered_analysis")

    print(f"\n{'='*80}")
    print(f"FIXING EVENT DATES FOR {ticker}")
    print(f"{'='*80}")

    # Load stock data to get date range
    stock_file = data_dir / f"{ticker}_stock_data.csv"
    stock_df = pd.read_csv(stock_file)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)
    stock_min = stock_df['Date'].min()
    stock_max = stock_df['Date'].max()

    print(f"\n📊 Stock data date range:")
    print(f"  Min: {stock_min.date()}")
    print(f"  Max: {stock_max.date()}")
    print(f"  Total days: {len(stock_df)}")

    # Load event dates
    event_file = filtered_dir / f"{ticker}_event_dates.csv"
    event_df = pd.read_csv(event_file)
    event_df['Date'] = pd.to_datetime(event_df['Date'])

    print(f"\n📅 Original event dates:")
    print(f"  Min: {event_df['Date'].min().date()}")
    print(f"  Max: {event_df['Date'].max().date()}")
    print(f"  Total events: {len(event_df)}")

    # Filter events to stock date range
    event_df_filtered = event_df[
        (event_df['Date'] >= stock_min) &
        (event_df['Date'] <= stock_max)
    ].copy()

    print(f"\n✅ Filtered event dates:")
    print(f"  Min: {event_df_filtered['Date'].min().date()}")
    print(f"  Max: {event_df_filtered['Date'].max().date()}")
    print(f"  Total events: {len(event_df_filtered)}")
    print(f"  Removed: {len(event_df) - len(event_df_filtered)} events")

    # Calculate event density
    trading_days = len(stock_df)
    event_days = len(event_df_filtered)
    density = event_days / trading_days * 100

    print(f"\n📈 Event density:")
    print(f"  {event_days} events / {trading_days} trading days = {density:.1f}%")

    if density > 80:
        print(f"  ⚠️  WARNING: Very high event density ({density:.1f}%)")
        print(f"  This may cause issues with beta estimation")
        print(f"  Consider using stricter filtering criteria")

    # Save corrected event dates
    output_file = filtered_dir / f"{ticker}_event_dates_corrected.csv"
    event_df_filtered.to_csv(output_file, index=False)
    print(f"\n💾 Saved corrected event dates to: {output_file.name}")

    return event_df_filtered


def main():
    """Fix event dates for both tickers"""
    print("\n" + "="*80)
    print("EVENT DATES CORRECTION TOOL")
    print("="*80)
    print("\nThis tool aligns event dates with available stock data")

    for ticker in ['TSLA', 'AAPL']:
        fix_event_dates(ticker)

    print("\n" + "="*80)
    print("✅ CORRECTION COMPLETE")
    print("="*80)
    print("\nNext step: Update the event study to use *_event_dates_corrected.csv files")
    print("="*80)


if __name__ == "__main__":
    main()
