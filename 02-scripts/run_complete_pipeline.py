"""
Complete End-to-End Pipeline
Downloads data from Yahoo Finance and Finnhub, then runs Phase 1 analysis
"""

import sys
from pathlib import Path
import warnings
import importlib
warnings.filterwarnings('ignore')

# Import data acquisition modules using importlib to handle numeric prefixes
data_acq_module = importlib.import_module('00_data_acquisition')
marketaux_module = importlib.import_module('00b_marketaux_news')
main_analysis_module = importlib.import_module('05_main_analysis')

DataAcquisition = data_acq_module.DataAcquisition
MarketauxNewsAcquisition = marketaux_module.MarketauxNewsAcquisition
Phase1Analysis = main_analysis_module.Phase1Analysis


class CompletePipeline:
    """End-to-end pipeline from data acquisition to analysis"""

    def __init__(self,
                 ticker: str,
                 start_date: str,
                 end_date: str = None,
                 sector_ticker: str = None,
                 marketaux_api_key: str = None,
                 data_dir: str = "../01-data",
                 output_dir: str = "../03-output"):
        """
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date (default: today)
        sector_ticker : str, optional
            Sector ETF ticker (e.g., 'XLK' for Tech)
        marketaux_api_key : str, optional
            Marketaux API key (will try .env if not provided)
        data_dir : str
            Data directory
        output_dir : str
            Output directory
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.sector_ticker = sector_ticker
        self.marketaux_api_key = marketaux_api_key
        self.data_dir = data_dir
        self.output_dir = output_dir

        # File paths
        self.stock_file = f"{ticker}_stock_data.csv"
        self.ff_file = "fama_french_factors.csv"
        self.sector_file = f"{sector_ticker}_sector_factor.csv" if sector_ticker else None
        self.news_file = "news_dates.csv"

    def run_full_pipeline(self, skip_data_download: bool = False):
        """
        Run complete pipeline from data acquisition to analysis

        Parameters:
        -----------
        skip_data_download : bool
            If True, skip data download and use existing files
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "COMPLETE NEWS-STOCK ANALYSIS PIPELINE")
        print("=" * 80)
        print(f"\nTicker: {self.ticker}")
        print(f"Period: {self.start_date} to {self.end_date or 'today'}")
        print(f"Sector ETF: {self.sector_ticker or 'None'}")
        print("=" * 80)

        try:
            # STAGE 1: Data Acquisition
            if not skip_data_download:
                self._stage1_data_acquisition()
            else:
                print("\n[STAGE 1] SKIPPING DATA DOWNLOAD (using existing files)")

            # STAGE 2: Phase 1 Analysis
            self._stage2_phase1_analysis()

            # STAGE 3: Summary
            self._stage3_summary()

        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _stage1_data_acquisition(self):
        """Stage 1: Download all required data"""
        print("\n" + "=" * 80)
        print("[STAGE 1] DATA ACQUISITION")
        print("=" * 80)

        # 1.1: Yahoo Finance data
        print("\n[1.1] Downloading stock and Fama-French data from Yahoo Finance...")
        print("-" * 80)

        stock_acquirer = DataAcquisition(output_dir=self.data_dir)

        # Download stock data
        stock_data = stock_acquirer.download_stock_data(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            save=True
        )

        # Download Fama-French factors
        ff_data = stock_acquirer.download_fama_french_factors(
            start_date=self.start_date,
            end_date=self.end_date,
            save=True
        )

        # Download sector factor if specified
        if self.sector_ticker:
            sector_data = stock_acquirer.download_sector_etf(
                sector_ticker=self.sector_ticker,
                start_date=self.start_date,
                end_date=self.end_date,
                save=True
            )

        # 1.2: Marketaux news data
        print("\n[1.2] Downloading news data from Marketaux...")
        print("-" * 80)

        news_acquirer = MarketauxNewsAcquisition(
            api_key=self.marketaux_api_key,
            output_dir=self.data_dir
        )

        news_data = news_acquirer.download_news(
            ticker=self.ticker,
            start_date=self.start_date,
            end_date=self.end_date,
            save=True
        )

        if len(news_data) == 0:
            raise ValueError(
                f"No news data found for {self.ticker} between {self.start_date} and {self.end_date}. "
                "Please check:\n"
                "  1. Your Marketaux API key is valid\n"
                "  2. The ticker symbol is correct\n"
                "  3. The date range has news coverage"
            )

        print("\n✅ STAGE 1 COMPLETE: All data downloaded")

    def _stage2_phase1_analysis(self):
        """Stage 2: Run Phase 1 event study analysis"""
        print("\n" + "=" * 80)
        print("[STAGE 2] PHASE 1 ANALYSIS - EVENT STUDY")
        print("=" * 80)

        analysis = Phase1Analysis(
            stock_file=self.stock_file,
            news_file=self.news_file,
            ff_file=self.ff_file,
            sector_file=self.sector_file,
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )

        # Run complete analysis
        self.summary = analysis.run_complete_analysis()

        print("\n✅ STAGE 2 COMPLETE: Analysis finished")

    def _stage3_summary(self):
        """Stage 3: Print final summary"""
        print("\n" + "=" * 80)
        print("[STAGE 3] PIPELINE SUMMARY")
        print("=" * 80)

        print(f"\n📊 Analysis Results for {self.ticker}:")
        print(f"   Period: {self.start_date} to {self.end_date or 'today'}")
        print(f"\n   News Days: {self.summary['news_days']}")
        print(f"   Non-News Days: {self.summary['non_news_days']}")
        print(f"\n   Mean AR (News Days): {self.summary['mean_ar_news']*100:.2f}%")
        print(f"   Mean AR (Non-News Days): {self.summary['mean_ar_non_news']*100:.2f}%")
        print(f"\n   Model R²: {self.summary['avg_r_squared']:.3f}")
        print(f"   Significant Tests: {self.summary['significant_tests']}/{self.summary['total_tests']}")

        # Success evaluation
        print("\n" + "=" * 80)
        if self.summary['significant_tests'] >= 3:
            print("✅ SUCCESS: Strong evidence that news impacts stock prices!")
        elif self.summary['significant_tests'] >= 2:
            print("⚠️  PARTIAL SUCCESS: Some evidence of news impact")
        else:
            print("❌ INCONCLUSIVE: Weak evidence of news impact")

        print("\n📁 Output Files:")
        print(f"   Data: {Path(self.data_dir).absolute()}")
        print(f"   Results: {Path(self.output_dir).absolute()}")

        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 80)


def run_pipeline(ticker: str,
                start_date: str,
                end_date: str = None,
                sector_ticker: str = None,
                marketaux_api_key: str = None,
                skip_data_download: bool = False):
    """
    Convenience function to run complete pipeline

    Parameters:
    -----------
    ticker : str
        Stock ticker (e.g., 'AAPL', 'MSFT', 'GOOGL')
    start_date : str
        Start date 'YYYY-MM-DD'
    end_date : str, optional
        End date (default: today)
    sector_ticker : str, optional
        Sector ETF ticker (e.g., 'XLK', 'XLF')
    marketaux_api_key : str, optional
        Marketaux API key (loads from .env if not provided)
    skip_data_download : bool
        Skip data download and use existing files

    Example:
    --------
    # Full pipeline with data download
    run_pipeline('AAPL', '2024-01-01', sector_ticker='XLK')

    # Analysis only (skip data download)
    run_pipeline('AAPL', '2024-01-01', skip_data_download=True)
    """
    pipeline = CompletePipeline(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        sector_ticker=sector_ticker,
        finnhub_api_key=finnhub_api_key
    )

    pipeline.run_full_pipeline(skip_data_download=skip_data_download)


if __name__ == "__main__":
    """
    Example usage:

    # Run from command line:
    python run_complete_pipeline.py

    # Or import and use:
    from run_complete_pipeline import run_pipeline
    run_pipeline('AAPL', '2020-01-01', sector_ticker='XLK')
    """

    print("\n" + "=" * 80)
    print("Complete News-Stock Prediction Pipeline")
    print("=" * 80)
    print("\nTo run the pipeline, use one of these methods:\n")
    print("1. Python script:")
    print("   from run_complete_pipeline import run_pipeline")
    print("   run_pipeline('AAPL', '2020-01-01', sector_ticker='XLK')\n")
    print("2. Modify this file and add your parameters at the bottom:\n")
    print("   # Example configuration")
    print("   TICKER = 'AAPL'")
    print("   START_DATE = '2020-01-01'")
    print("   SECTOR_TICKER = 'XLK'  # Technology sector")
    print("   ")
    print("   run_pipeline(TICKER, START_DATE, sector_ticker=SECTOR_TICKER)")
    print("\n" + "=" * 80)

    # Uncomment and configure to run:
    # run_pipeline('AAPL', '2020-01-01', sector_ticker='XLK')