"""
Abnormal Returns Calculation Module
Calculates expected returns using estimated betas and computes abnormal returns
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple


class AbnormalReturnsCalculator:
    """Calculates abnormal returns using estimated factor models"""

    def __init__(self):
        pass

    def calculate_expected_returns(self,
                                   data: pd.DataFrame,
                                   beta_df: pd.DataFrame,
                                   factor_cols: List[str]) -> pd.Series:
        """
        Calculate expected returns using estimated betas

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with factor returns
        beta_df : pd.DataFrame
            DataFrame with estimated betas and alpha
        factor_cols : List[str]
            List of factor column names

        Returns:
        --------
        pd.Series of expected returns
        """
        # Align data
        common_dates = data.index.intersection(beta_df.index)
        data_aligned = data.loc[common_dates]
        beta_aligned = beta_df.loc[common_dates]

        # Calculate expected return: Alpha + Sum(Beta_i * Factor_i)
        expected_returns = beta_aligned['Alpha'].copy()

        for factor in factor_cols:
            beta_col = f'Beta_{factor}'
            if beta_col in beta_aligned.columns and factor in data_aligned.columns:
                expected_returns += beta_aligned[beta_col] * data_aligned[factor]

        return expected_returns

    def calculate_abnormal_returns(self,
                                   data: pd.DataFrame,
                                   beta_df: pd.DataFrame,
                                   factor_cols: List[str]) -> pd.DataFrame:
        """
        Calculate abnormal returns (AR = Actual - Expected)

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with actual returns
        beta_df : pd.DataFrame
            DataFrame with estimated betas
        factor_cols : List[str]
            List of factor column names

        Returns:
        --------
        pd.DataFrame with actual, expected, and abnormal returns
        """
        # Calculate expected returns
        expected_returns = self.calculate_expected_returns(data, beta_df, factor_cols)

        # Create results DataFrame
        results = pd.DataFrame(index=expected_returns.index)
        results['Actual_Return'] = data.loc[expected_returns.index, 'Return']
        results['Expected_Return'] = expected_returns
        results['Abnormal_Return'] = results['Actual_Return'] - results['Expected_Return']

        # Add R-squared from model
        results['R_squared'] = beta_df.loc[expected_returns.index, 'R_squared']

        return results

    def calculate_cumulative_abnormal_returns(self,
                                            ar_df: pd.DataFrame,
                                            news_dates: pd.Series,
                                            window: Tuple[int, int] = (-1, 2)) -> pd.DataFrame:
        """
        Calculate Cumulative Abnormal Returns (CAR) around news events

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns
        news_dates : pd.Series
            Series of news event dates
        window : tuple
            Event window as (days_before, days_after)

        Returns:
        --------
        pd.DataFrame with CAR for each news event
        """
        car_results = []

        for news_date in news_dates:
            window_start = news_date + pd.Timedelta(days=window[0])
            window_end = news_date + pd.Timedelta(days=window[1])

            # Get abnormal returns in window
            window_mask = (ar_df.index >= window_start) & (ar_df.index <= window_end)
            window_ar = ar_df.loc[window_mask, 'Abnormal_Return']

            if len(window_ar) > 0:
                car_results.append({
                    'News_Date': news_date,
                    'Window_Start': window_start,
                    'Window_End': window_end,
                    'CAR': window_ar.sum(),
                    'N_days': len(window_ar),
                    'Mean_AR': window_ar.mean(),
                    'Std_AR': window_ar.std()
                })

        return pd.DataFrame(car_results)

    def tag_news_days(self,
                     ar_df: pd.DataFrame,
                     news_dates: pd.Series,
                     window: Tuple[int, int] = (0, 0)) -> pd.DataFrame:
        """
        Tag dates as news days or non-news days

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns
        news_dates : pd.Series
            Series of news event dates
        window : tuple
            Event window to consider as "news day"

        Returns:
        --------
        pd.DataFrame with News_Day indicator
        """
        result = ar_df.copy()
        result['News_Day'] = False

        for news_date in news_dates:
            window_start = news_date + pd.Timedelta(days=window[0])
            window_end = news_date + pd.Timedelta(days=window[1])
            mask = (result.index >= window_start) & (result.index <= window_end)
            result.loc[mask, 'News_Day'] = True

        return result

    def calculate_ar_statistics(self,
                                ar_df: pd.DataFrame,
                                news_indicator: str = 'News_Day') -> pd.DataFrame:
        """
        Calculate summary statistics for abnormal returns by news/non-news days

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns and news day indicator
        news_indicator : str
            Column name for news day indicator

        Returns:
        --------
        pd.DataFrame with statistics
        """
        stats = []

        for is_news_day in [True, False]:
            subset = ar_df[ar_df[news_indicator] == is_news_day]['Abnormal_Return'].dropna()

            if len(subset) > 0:
                stats.append({
                    'Category': 'News Days' if is_news_day else 'Non-News Days',
                    'N': len(subset),
                    'Mean': subset.mean(),
                    'Std': subset.std(),
                    'Min': subset.min(),
                    'Max': subset.max(),
                    'Median': subset.median(),
                    '25th_Percentile': subset.quantile(0.25),
                    '75th_Percentile': subset.quantile(0.75)
                })

        # Return DataFrame with proper columns even if empty
        if not stats:
            return pd.DataFrame(columns=['Category', 'N', 'Mean', 'Std', 'Min', 'Max',
                                        'Median', '25th_Percentile', '75th_Percentile'])
        return pd.DataFrame(stats)


def calculate_abnormal_returns(data: pd.DataFrame,
                               beta_df: pd.DataFrame,
                               news_dates: pd.Series,
                               factor_cols: List[str] = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']) -> pd.DataFrame:
    """
    Convenience function to calculate abnormal returns

    Returns:
    --------
    pd.DataFrame with abnormal returns and news day tags
    """
    calculator = AbnormalReturnsCalculator()
    ar_df = calculator.calculate_abnormal_returns(data, beta_df, factor_cols)
    ar_df = calculator.tag_news_days(ar_df, news_dates, window=(0, 0))
    return ar_df


if __name__ == "__main__":
    print("Abnormal Returns Calculation Module")
    print("Computes AR = Actual Return - Expected Return")
    print("Also calculates Cumulative Abnormal Returns (CAR) around news events")
