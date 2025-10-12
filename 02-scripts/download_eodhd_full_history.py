"""
Download Complete EODHD News History
Downloads all historical news year-by-year to avoid timeouts
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Import the EODHD news module
sys.path.append(str(Path(__file__).parent))
from importlib import import_module
eodhd_module = import_module('00c_eodhd_news')
EODHDNewsAcquisition = eodhd_module.EODHDNewsAcquisition


def download_news_by_year(downloader, ticker, start_year, end_year):
    """
    Download news year by year to avoid timeouts

    Parameters:
    -----------
    downloader : EODHDNewsAcquisition
        Initialized downloader instance
    ticker : str
        Stock ticker
    start_year : int
        Start year
    end_year : int
        End year (inclusive)
    """
    all_data = []

    print(f"\n{'='*70}")
    print(f"Downloading {ticker} news from {start_year} to {end_year}")
    print(f"{'='*70}\n")

    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"

        # For last year, use specific end date
        if year == end_year:
            if ticker == 'AAPL':
                end_date = '2024-01-31'
            elif ticker == 'TSLA':
                end_date = '2025-10-08'
            else:
                end_date = f"{year}-12-31"
        else:
            end_date = f"{year}-12-31"

        print(f"\n[Year {year}] {ticker}: {start_date} to {end_date}")
        print("-" * 70)

        try:
            # Download for this year (don't save yet)
            df = downloader.download_news(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                limit=1000,
                save=False
            )

            if not df.empty:
                all_data.append(df)
                print(f"  ✓ Year {year}: {len(df)} articles collected")
            else:
                print(f"  ⚠ Year {year}: No articles found")

        except Exception as e:
            print(f"  ✗ Year {year}: Error - {e}")
            continue

    # Combine all years
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates based on link
        original_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['link'], keep='first')
        dedupe_len = len(combined_df)

        # Sort by date
        combined_df = combined_df.sort_values('date')

        print(f"\n{'='*70}")
        print(f"COMBINED RESULTS FOR {ticker}")
        print(f"{'='*70}")
        print(f"  Total articles: {dedupe_len:,} (removed {original_len - dedupe_len:,} duplicates)")
        print(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        print(f"  Unique dates: {combined_df['date'].dt.date.nunique():,}")

        # Save combined file
        output_file = downloader.output_dir / f"{ticker}_eodhd_news.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved to {output_file}")

        return combined_df
    else:
        print(f"\n  ⚠ No data collected for {ticker}")
        return pd.DataFrame()


def main():
    """Download all historical news for AAPL and TSLA"""

    print("\n" + "="*70)
    print(" " * 15 + "EODHD FULL HISTORICAL NEWS DOWNLOAD")
    print("="*70)
    print("\nThis will download ALL historical news:")
    print("  - AAPL: 2020-2024 (year by year)")
    print("  - TSLA: 2020-2025 (year by year)")
    print("\nUsing your paid EODHD plan for complete historical access")
    print("="*70)

    try:
        # Initialize downloader
        downloader = EODHDNewsAcquisition(output_dir='../01-data')

        # Download AAPL news (2020-2024)
        print("\n\n" + "🍎" * 35)
        aapl_df = download_news_by_year(
            downloader=downloader,
            ticker='AAPL',
            start_year=2020,
            end_year=2024
        )

        # Download TSLA news (2020-2025)
        print("\n\n" + "⚡" * 35)
        tsla_df = download_news_by_year(
            downloader=downloader,
            ticker='TSLA',
            start_year=2020,
            end_year=2025
        )

        # Final summary
        print("\n" + "="*70)
        print(" " * 25 + "DOWNLOAD COMPLETE!")
        print("="*70)
        print(f"\nFinal Summary:")
        if not aapl_df.empty:
            print(f"  AAPL: {len(aapl_df):,} articles")
            print(f"        Date range: {aapl_df['date'].min().date()} to {aapl_df['date'].max().date()}")
        if not tsla_df.empty:
            print(f"  TSLA: {len(tsla_df):,} articles")
            print(f"        Date range: {tsla_df['date'].min().date()} to {tsla_df['date'].max().date()}")
        print("\n" + "="*70)
        print("✅ All data saved to: ../01-data/")
        print("="*70)

    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure EODHD_API_KEY is set in .env file")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
