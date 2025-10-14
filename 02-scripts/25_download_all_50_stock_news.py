"""
DOWNLOAD NEWS FOR ALL 50 STOCKS
=================================

Downloads news data from EODHD for all 50 stocks
"""

import sys
from pathlib import Path
import time

# Import configuration
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

config = import_module('21_expanded_50_stock_config')
news_module = import_module('00c_eodhd_news')

EXPANDED_STOCKS = config.EXPANDED_STOCKS
EODHDNewsAcquisition = news_module.EODHDNewsAcquisition

DATA_DIR = Path("../01-data")
# Date range - Updated to 2021-2025 based on EODHD news data availability analysis
START_DATE = '2021-01-01'
END_DATE = '2025-07-31'

def check_existing_news():
    """Check which stocks already have news data"""
    existing = []
    missing = []

    for ticker in EXPANDED_STOCKS.keys():
        news_file = DATA_DIR / f"{ticker}_eodhd_news.csv"
        if news_file.exists():
            existing.append(ticker)
        else:
            missing.append(ticker)

    return existing, missing

def main():
    print("="*80)
    print("DOWNLOADING NEWS FOR 50-STOCK CONFIGURATION")
    print("="*80)

    existing, missing = check_existing_news()

    print(f"\n✅ Stocks with existing news data: {len(existing)}")
    print(f"❌ Stocks needing news data: {len(missing)}")

    if not missing:
        print("\n✅ All stocks already have news data!")
        return

    print(f"\nMissing news for {len(missing)} stocks:")
    # Group by sector
    missing_by_sector = {}
    for ticker in missing:
        sector = EXPANDED_STOCKS[ticker]['sector']
        if sector not in missing_by_sector:
            missing_by_sector[sector] = []
        missing_by_sector[sector].append(ticker)

    for sector in sorted(missing_by_sector.keys()):
        tickers = missing_by_sector[sector]
        print(f"  {sector}: {', '.join(tickers)}")

    print("\nStarting downloads...")

    # Initialize downloader
    try:
        downloader = EODHDNewsAcquisition(output_dir=str(DATA_DIR))
    except Exception as e:
        print(f"\n❌ Error initializing downloader: {e}")
        return

    successful = []
    failed = []

    for i, ticker in enumerate(missing, 1):
        sector = EXPANDED_STOCKS[ticker]['sector']
        name = EXPANDED_STOCKS[ticker]['name']

        print(f"\n[{i}/{len(missing)}] {ticker} - {name} ({sector})")

        try:
            df = downloader.download_news(
                ticker=ticker,
                start_date=START_DATE,
                end_date=END_DATE,
                save=True
            )

            if df is not None and len(df) > 0:
                successful.append(ticker)
                print(f"  ✅ Downloaded {len(df)} articles")
            else:
                print(f"  ⚠️ No articles found")
                failed.append(ticker)

        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
            failed.append(ticker)

        # Rate limiting (EODHD has API limits)
        if i < len(missing):
            time.sleep(3)  # 3 seconds between requests

    # Summary
    print("\n" + "="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"✅ Successful: {len(successful)}/{len(missing)}")

    if successful:
        print(f"\nSuccessfully downloaded:")
        for ticker in successful:
            print(f"  {ticker} - {EXPANDED_STOCKS[ticker]['name']}")

    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for ticker in failed:
            print(f"  {ticker} - {EXPANDED_STOCKS[ticker]['name']}")

    # Final check
    existing, missing = check_existing_news()
    print(f"\n📊 Final status: {len(existing)}/50 stocks have news data")

    if missing:
        print(f"Still missing ({len(missing)}): {', '.join(sorted(missing))}")

if __name__ == "__main__":
    main()
