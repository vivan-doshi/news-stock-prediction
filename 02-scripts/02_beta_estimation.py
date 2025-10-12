"""
Beta Estimation Module using Fama-French 5-Factor Model
Estimates factor loadings using rolling window OLS regression
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from scipy import stats


class BetaEstimator:
    """Estimates factor betas using rolling window regression"""

    def __init__(self,
                 window_size: int = 126,
                 min_periods: int = 100):
        """
        Parameters:
        -----------
        window_size : int
            Number of trading days for beta estimation (default 126 = ~6 months)
        min_periods : int
            Minimum number of observations required for estimation
        """
        self.window_size = window_size
        self.min_periods = min_periods

    def estimate_betas_ols(self,
                          X: np.ndarray,
                          y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Estimate betas using OLS regression: β = (X'X)^(-1)X'y

        Parameters:
        -----------
        X : np.ndarray
            Factor returns matrix (n_observations x n_factors)
        y : np.ndarray
            Excess stock returns (n_observations,)

        Returns:
        --------
        Tuple of (betas, r_squared, residuals)
        """
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS formula: β = (X'X)^(-1)X'y
        try:
            betas = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

            # Calculate R-squared
            y_pred = X_with_intercept @ betas
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return betas, r_squared, residuals

        except np.linalg.LinAlgError:
            # Return NaN if matrix is singular
            return np.full(X_with_intercept.shape[1], np.nan), np.nan, np.full(len(y), np.nan)

    def rolling_beta_estimation(self,
                                data: pd.DataFrame,
                                factor_cols: List[str],
                                exclude_dates: Optional[pd.Series] = None,
                                event_window: Tuple[int, int] = (-1, 2)) -> pd.DataFrame:
        """
        Estimate betas using rolling window, excluding news event days

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with Excess_Return and factor columns
        factor_cols : List[str]
            List of factor column names (e.g., ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        exclude_dates : pd.Series, optional
            News dates to exclude from estimation
        event_window : tuple
            Window around news dates to exclude (days_before, days_after)

        Returns:
        --------
        pd.DataFrame with estimated betas for each date
        """
        results = []

        # Create exclusion mask
        exclusion_mask = self._create_exclusion_mask(data.index, exclude_dates, event_window)

        for i, date in enumerate(data.index):
            if i < self.min_periods:
                # Not enough data yet
                results.append(self._create_nan_result(date, factor_cols))
                continue

            # Get window data
            start_idx = max(0, i - self.window_size)
            window_data = data.iloc[start_idx:i].copy()

            # Exclude news event days from estimation
            if exclusion_mask is not None:
                window_mask = exclusion_mask.loc[window_data.index]
                window_data = window_data[~window_mask]

            if len(window_data) < self.min_periods:
                results.append(self._create_nan_result(date, factor_cols))
                continue

            # Prepare X and y
            X = window_data[factor_cols].values
            y = window_data['Excess_Return'].values

            # Estimate betas
            betas, r_squared, residuals = self.estimate_betas_ols(X, y)

            # Store results
            result = {
                'Date': date,
                'Alpha': betas[0],
                'R_squared': r_squared,
                'N_obs': len(window_data),
                'Residual_Std': np.std(residuals) if not np.isnan(residuals).all() else np.nan
            }

            for j, factor in enumerate(factor_cols):
                result[f'Beta_{factor}'] = betas[j + 1]

            results.append(result)

        return pd.DataFrame(results).set_index('Date')

    def _create_exclusion_mask(self,
                               all_dates: pd.DatetimeIndex,
                               exclude_dates: Optional[pd.Series],
                               event_window: Tuple[int, int]) -> Optional[pd.Series]:
        """Create boolean mask for dates to exclude from estimation"""
        if exclude_dates is None or len(exclude_dates) == 0:
            return None

        mask = pd.Series(False, index=all_dates)

        for news_date in exclude_dates:
            window_start = news_date + pd.Timedelta(days=event_window[0])
            window_end = news_date + pd.Timedelta(days=event_window[1])
            mask.loc[window_start:window_end] = True

        return mask

    def _create_nan_result(self, date: pd.Timestamp, factor_cols: List[str]) -> dict:
        """Create NaN result for insufficient data"""
        result = {
            'Date': date,
            'Alpha': np.nan,
            'R_squared': np.nan,
            'N_obs': 0,
            'Residual_Std': np.nan
        }
        for factor in factor_cols:
            result[f'Beta_{factor}'] = np.nan
        return result

    def calculate_beta_stability(self, beta_df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
        """
        Calculate beta stability metrics (coefficient of variation)

        Parameters:
        -----------
        beta_df : pd.DataFrame
            DataFrame with estimated betas
        factor_cols : List[str]
            List of factor names

        Returns:
        --------
        pd.DataFrame with stability metrics
        """
        stability = {}

        for factor in factor_cols:
            beta_col = f'Beta_{factor}'
            if beta_col in beta_df.columns:
                betas = beta_df[beta_col].dropna()
                if len(betas) > 0:
                    stability[factor] = {
                        'Mean': betas.mean(),
                        'Std': betas.std(),
                        'CV': betas.std() / abs(betas.mean()) if betas.mean() != 0 else np.nan,
                        'Min': betas.min(),
                        'Max': betas.max()
                    }

        return pd.DataFrame(stability).T


def estimate_betas(data: pd.DataFrame,
                   factor_cols: List[str] = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
                   news_dates: Optional[pd.Series] = None,
                   window_size: int = 126) -> pd.DataFrame:
    """
    Convenience function to estimate betas

    Returns:
    --------
    pd.DataFrame with beta estimates for each date
    """
    estimator = BetaEstimator(window_size=window_size)
    return estimator.rolling_beta_estimation(data, factor_cols, news_dates)


if __name__ == "__main__":
    print("Beta Estimation Module")
    print("Uses Fama-French 5-factor model with rolling OLS regression")
    print("Excludes news event windows from estimation to avoid contamination")
