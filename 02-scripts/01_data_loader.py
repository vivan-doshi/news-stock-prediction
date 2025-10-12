"""
Data Loading Module for News-Stock Impact Analysis
Loads stock prices, news dates, and Fama-French factor data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """Handles loading and preprocessing of all required data"""

    def __init__(self, data_dir: str = "../01-data"):
        self.data_dir = Path(data_dir)

    def load_stock_data(self, filepath: str) -> pd.DataFrame:
        """
        Load stock price data

        Parameters:
        -----------
        filepath : str
            Path to stock data file (CSV format)
            Expected columns: Date, Close, Volume

        Returns:
        --------
        pd.DataFrame with Date index and stock data
        """
        full_path = self.data_dir / filepath
        df = pd.read_csv(full_path)

        # Convert Date column to datetime, handling timezone info
        # Normalize to date only (no time component) for join compatibility
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None).dt.normalize()
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Calculate returns
        df['Return'] = df['Close'].pct_change()

        return df

    def load_news_dates(self, filepath: str) -> pd.Series:
        """
        Load news event dates

        Parameters:
        -----------
        filepath : str
            Path to news dates file (CSV format)
            Expected columns: Date

        Returns:
        --------
        pd.Series of news dates
        """
        full_path = self.data_dir / filepath
        df = pd.read_csv(full_path, parse_dates=['Date'])
        news_dates = pd.to_datetime(df['Date'])
        return news_dates

    def load_fama_french_factors(self, filepath: str) -> pd.DataFrame:
        """
        Load Fama-French 5-factor data

        Parameters:
        -----------
        filepath : str
            Path to Fama-French factors file (CSV format)
            Expected columns: Date, Mkt-RF, SMB, HML, RMW, CMA, RF

        Returns:
        --------
        pd.DataFrame with Date index and factor returns
        """
        full_path = self.data_dir / filepath
        df = pd.read_csv(full_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        # Convert percentages to decimals if needed
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for col in factor_cols:
            if df[col].abs().mean() > 1:  # Likely in percentage form
                df[col] = df[col] / 100

        return df

    def load_sector_factor(self, filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load sector-specific factor returns

        Parameters:
        -----------
        filepath : str, optional
            Path to sector factor file

        Returns:
        --------
        pd.DataFrame with Date index and sector returns, or None
        """
        if filepath is None:
            return None

        full_path = self.data_dir / filepath
        df = pd.read_csv(full_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        if df['Sector_Return'].abs().mean() > 1:
            df['Sector_Return'] = df['Sector_Return'] / 100

        return df

    def merge_data(self,
                   stock_df: pd.DataFrame,
                   ff_df: pd.DataFrame,
                   sector_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge stock data with Fama-French factors

        Parameters:
        -----------
        stock_df : pd.DataFrame
            Stock price data with returns
        ff_df : pd.DataFrame
            Fama-French factors
        sector_df : pd.DataFrame, optional
            Sector factor data

        Returns:
        --------
        pd.DataFrame with merged data
        """
        merged = stock_df.join(ff_df, how='inner')

        if sector_df is not None:
            merged = merged.join(sector_df, how='inner')

        # Calculate excess returns
        merged['Excess_Return'] = merged['Return'] - merged['RF']

        # Drop NaN rows
        merged.dropna(inplace=True)

        return merged

    def create_event_windows(self,
                            news_dates: pd.Series,
                            window: Tuple[int, int] = (-1, 2)) -> pd.DataFrame:
        """
        Create event window indicators

        Parameters:
        -----------
        news_dates : pd.Series
            Series of news event dates
        window : tuple
            Event window as (days_before, days_after)

        Returns:
        --------
        pd.DataFrame with news dates and event windows
        """
        event_data = []

        for news_date in news_dates:
            event_data.append({
                'News_Date': news_date,
                'Window_Start': pd.Timestamp(news_date) + pd.Timedelta(days=window[0]),
                'Window_End': pd.Timestamp(news_date) + pd.Timedelta(days=window[1])
            })

        return pd.DataFrame(event_data)


def load_all_data(stock_file: str,
                  news_file: str,
                  ff_file: str,
                  sector_file: Optional[str] = None,
                  data_dir: str = "../01-data") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load all required data

    Returns:
    --------
    Tuple of (merged_data, news_dates)
    """
    loader = DataLoader(data_dir)

    stock_df = loader.load_stock_data(stock_file)
    news_dates = loader.load_news_dates(news_file)
    ff_df = loader.load_fama_french_factors(ff_file)
    sector_df = loader.load_sector_factor(sector_file) if sector_file else None

    merged_data = loader.merge_data(stock_df, ff_df, sector_df)

    return merged_data, news_dates


if __name__ == "__main__":
    # Example usage
    print("Data Loader Module")
    print("Use load_all_data() to load stock, news, and factor data")
