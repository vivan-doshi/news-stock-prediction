"""
Data Acquisition Module
Downloads stock data from Yahoo Finance and Fama-French factors from Kenneth French Data Library
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import requests
from io import BytesIO
import zipfile
import warnings
warnings.filterwarnings('ignore')


class DataAcquisition:
    """Handles data download from Yahoo Finance and Fama-French library"""

    def __init__(self, output_dir: str = "../01-data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def download_stock_data(self,
                           ticker: str,
                           start_date: str,
                           end_date: Optional[str] = None,
                           save: bool = True) -> pd.DataFrame:
        """
        Download stock price data from Yahoo Finance

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        save : bool
            Save to CSV file (default: True)

        Returns:
        --------
        pd.DataFrame with stock price data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading {ticker} data from {start_date} to {end_date}...")

        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)

        if len(df) == 0:
            raise ValueError(f"No data found for {ticker}")

        # Clean and prepare data
        df.index = pd.to_datetime(df.index).date
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # Select relevant columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        print(f"  ✓ Downloaded {len(df)} days of data")

        if save:
            filename = self.output_dir / f"{ticker}_stock_data.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ Saved to {filename}")

        return df

    def download_fama_french_factors(self,
                                     start_date: str,
                                     end_date: Optional[str] = None,
                                     save: bool = True) -> pd.DataFrame:
        """
        Download Fama-French 5-factor data from Kenneth French Data Library

        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (default: today)
        save : bool
            Save to CSV file (default: True)

        Returns:
        --------
        pd.DataFrame with Fama-French factors
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading Fama-French 5-factor data from {start_date} to {end_date}...")

        # URL for Fama-French 5 Factors (daily)
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

        try:
            # Download and extract
            response = requests.get(url)
            zip_file = zipfile.ZipFile(BytesIO(response.content))
            csv_file = zip_file.namelist()[0]

            # Read CSV
            df = pd.read_csv(zip_file.open(csv_file), skiprows=3)

            # Find where the data ends (usually indicated by blank lines or annual data)
            df = df[df.iloc[:, 0].astype(str).str.isdigit()]

            # Rename columns
            df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']

            # Convert date format from YYYYMMDD to datetime
            df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d')

            # Convert factor returns from percentage to decimal
            factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            for col in factor_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100

            # Filter date range
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

            # Remove any NaN rows
            df.dropna(inplace=True)

            print(f"  ✓ Downloaded {len(df)} days of factor data")

            if save:
                filename = self.output_dir / "fama_french_factors.csv"
                df.to_csv(filename, index=False)
                print(f"  ✓ Saved to {filename}")

            return df

        except Exception as e:
            print(f"  ✗ Error downloading Fama-French factors: {e}")
            print("  ℹ Attempting alternative method...")
            return self._download_ff_alternative(start_date, end_date, save)

    def _download_ff_alternative(self,
                                 start_date: str,
                                 end_date: str,
                                 save: bool = True) -> pd.DataFrame:
        """
        Alternative method to get Fama-French factors using pandas_datareader

        Note: Requires pandas_datareader to be installed
        """
        try:
            import pandas_datareader as pdr

            print("  Using pandas_datareader for Fama-French data...")

            # Download from Kenneth French data library
            ff5 = pdr.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench',
                                start=start_date, end=end_date)[0]

            # Reset index and convert to proper format
            ff5.reset_index(inplace=True)
            ff5.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            ff5['Date'] = pd.to_datetime(ff5['Date'])

            # Convert from percentage to decimal
            factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            for col in factor_cols:
                ff5[col] = ff5[col] / 100

            print(f"  ✓ Downloaded {len(ff5)} days of factor data")

            if save:
                filename = self.output_dir / "fama_french_factors.csv"
                ff5.to_csv(filename, index=False)
                print(f"  ✓ Saved to {filename}")

            return ff5

        except ImportError:
            print("  ✗ pandas_datareader not installed")
            print("  Install with: pip install pandas-datareader")
            raise

    def download_sector_etf(self,
                           sector_ticker: str,
                           start_date: str,
                           end_date: Optional[str] = None,
                           save: bool = True) -> pd.DataFrame:
        """
        Download sector ETF data as a proxy for sector factor

        Parameters:
        -----------
        sector_ticker : str
            Sector ETF ticker (e.g., 'XLK' for Technology, 'XLF' for Financials)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        save : bool
            Save to CSV file

        Returns:
        --------
        pd.DataFrame with sector returns
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Downloading sector ETF {sector_ticker} from {start_date} to {end_date}...")

        # Download sector ETF data
        sector = yf.Ticker(sector_ticker)
        df = sector.history(start=start_date, end=end_date)

        if len(df) == 0:
            raise ValueError(f"No data found for sector ETF {sector_ticker}")

        # Calculate returns
        df.index = pd.to_datetime(df.index).date
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sector_Return'] = df['Close'].pct_change()

        # Select relevant columns
        df = df[['Date', 'Sector_Return']].dropna()

        print(f"  ✓ Downloaded {len(df)} days of sector data")

        if save:
            filename = self.output_dir / f"{sector_ticker}_sector_factor.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ Saved to {filename}")

        return df

    def download_all_data(self,
                         ticker: str,
                         start_date: str,
                         end_date: Optional[str] = None,
                         sector_ticker: Optional[str] = None) -> dict:
        """
        Download all required data in one go

        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date
        sector_ticker : str, optional
            Sector ETF ticker for sector factor
        create_sample_news : bool
            Create sample news dates for testing

        Returns:
        --------
        dict with all downloaded DataFrames
        """
        print("=" * 70)
        print("DATA ACQUISITION PIPELINE")
        print("=" * 70)

        data = {}

        # 1. Download stock data
        print("\n[1/4] Downloading stock data...")
        data['stock'] = self.download_stock_data(ticker, start_date, end_date)

        # 2. Download Fama-French factors
        print("\n[2/4] Downloading Fama-French factors...")
        data['factors'] = self.download_fama_french_factors(start_date, end_date)

        # 3. Download sector factor (optional)
        if sector_ticker:
            print(f"\n[3/3] Downloading sector factor ({sector_ticker})...")
            data['sector'] = self.download_sector_etf(sector_ticker, start_date, end_date)
        else:
            print("\n[3/3] Skipping sector factor (not specified)")
            data['sector'] = None

        print("\n" + "=" * 70)
        print("DATA ACQUISITION COMPLETE!")
        print("=" * 70)
        print(f"\nFiles saved to: {self.output_dir.absolute()}")
        print("\nNote: Use 00b_finnhub_news.py to download news data separately")

        return data


def download_data(ticker: str = 'AAPL',
                 start_date: str = '2020-01-01',
                 end_date: Optional[str] = None,
                 sector_ticker: Optional[str] = 'XLK',
                 output_dir: str = '../01-data'):
    """
    Convenience function to download all data

    Example usage:
    --------------
    download_data('AAPL', '2020-01-01', sector_ticker='XLK')
    """
    acquirer = DataAcquisition(output_dir=output_dir)
    return acquirer.download_all_data(ticker, start_date, end_date,
                                     sector_ticker=sector_ticker)


if __name__ == "__main__":
    print("Data Acquisition Module")
    print("\nExample usage:")
    print("  from data_acquisition import download_data")
    print("  download_data('AAPL', '2020-01-01', sector_ticker='XLK')")
    print("\nCommon sector ETFs:")
    print("  XLK - Technology")
    print("  XLF - Financials")
    print("  XLE - Energy")
    print("  XLV - Healthcare")
    print("  XLY - Consumer Discretionary")
    print("  XLP - Consumer Staples")
    print("  XLI - Industrials")
    print("  XLB - Materials")
    print("  XLU - Utilities")
    print("  XLRE - Real Estate")
