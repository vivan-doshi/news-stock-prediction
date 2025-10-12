"""
Statistical Testing Module
Performs hypothesis tests to determine if news significantly impacts stock prices
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Tuple, Dict, List, Optional


class StatisticalTester:
    """Performs various statistical tests on abnormal returns"""

    def __init__(self, alpha: float = 0.05):
        """
        Parameters:
        -----------
        alpha : float
            Significance level for hypothesis tests (default 0.05)
        """
        self.alpha = alpha

    def one_sample_ttest(self, data: pd.Series, null_mean: float = 0.0) -> Dict:
        """
        One-sample t-test: H0: mean(AR) = null_mean

        Parameters:
        -----------
        data : pd.Series
            Abnormal returns
        null_mean : float
            Null hypothesis mean (default 0.0)

        Returns:
        --------
        Dict with test results
        """
        clean_data = data.dropna()

        if len(clean_data) == 0:
            return self._empty_result("One-Sample t-test")

        t_stat, p_value = stats.ttest_1samp(clean_data, null_mean)

        return {
            'Test': 'One-Sample t-test',
            'N': len(clean_data),
            'Mean': clean_data.mean(),
            'Std': clean_data.std(),
            'SE': clean_data.std() / np.sqrt(len(clean_data)),
            't_statistic': t_stat,
            'p_value': p_value,
            'Significant': p_value < self.alpha,
            'Null_Hypothesis': f'Mean = {null_mean}',
            'Alternative': 'Mean ≠ 0' if null_mean == 0 else f'Mean ≠ {null_mean}'
        }

    def two_sample_ttest(self,
                        group1: pd.Series,
                        group2: pd.Series,
                        equal_var: bool = False) -> Dict:
        """
        Two-sample t-test: Compare news days vs non-news days

        Parameters:
        -----------
        group1 : pd.Series
            Abnormal returns for group 1 (e.g., news days)
        group2 : pd.Series
            Abnormal returns for group 2 (e.g., non-news days)
        equal_var : bool
            Assume equal variance (default False for Welch's t-test)

        Returns:
        --------
        Dict with test results
        """
        clean1 = group1.dropna()
        clean2 = group2.dropna()

        if len(clean1) == 0 or len(clean2) == 0:
            return self._empty_result("Two-Sample t-test")

        t_stat, p_value = stats.ttest_ind(clean1, clean2, equal_var=equal_var)

        return {
            'Test': "Welch's t-test" if not equal_var else "Independent t-test",
            'N_group1': len(clean1),
            'N_group2': len(clean2),
            'Mean_group1': clean1.mean(),
            'Mean_group2': clean2.mean(),
            'Std_group1': clean1.std(),
            'Std_group2': clean2.std(),
            'Mean_Difference': clean1.mean() - clean2.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'Significant': p_value < self.alpha,
            'Null_Hypothesis': 'Mean1 = Mean2',
            'Alternative': 'Mean1 ≠ Mean2'
        }

    def variance_test(self, group1: pd.Series, group2: pd.Series) -> Dict:
        """
        F-test for equality of variances

        Parameters:
        -----------
        group1 : pd.Series
            Abnormal returns for group 1
        group2 : pd.Series
            Abnormal returns for group 2

        Returns:
        --------
        Dict with test results
        """
        clean1 = group1.dropna()
        clean2 = group2.dropna()

        if len(clean1) < 2 or len(clean2) < 2:
            return self._empty_result("F-test")

        var1 = clean1.var()
        var2 = clean2.var()
        f_stat = var1 / var2 if var2 > 0 else np.nan
        df1 = len(clean1) - 1
        df2 = len(clean2) - 1

        # Two-tailed p-value
        p_value = 2 * min(stats.f.cdf(f_stat, df1, df2),
                         1 - stats.f.cdf(f_stat, df1, df2))

        return {
            'Test': 'F-test (Variance)',
            'N_group1': len(clean1),
            'N_group2': len(clean2),
            'Var_group1': var1,
            'Var_group2': var2,
            'F_statistic': f_stat,
            'df1': df1,
            'df2': df2,
            'p_value': p_value,
            'Significant': p_value < self.alpha,
            'Null_Hypothesis': 'Var1 = Var2',
            'Alternative': 'Var1 ≠ Var2'
        }

    def regression_analysis(self,
                           ar_df: pd.DataFrame,
                           news_indicator: str = 'News_Day') -> Dict:
        """
        Regression analysis: AR = β0 + β1*News_Day + ε

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns and news indicator
        news_indicator : str
            Column name for news day indicator

        Returns:
        --------
        Dict with regression results
        """
        clean_df = ar_df[['Abnormal_Return', news_indicator]].dropna()

        if len(clean_df) < 2:
            return self._empty_result("Regression Analysis")

        # Check if we have enough variation in news indicator
        news_count = clean_df[news_indicator].sum()
        if news_count < 2 or news_count > len(clean_df) - 2:
            return self._empty_result("Regression Analysis - Insufficient news variation")

        X = clean_df[news_indicator].astype(int).values.reshape(-1, 1)
        y = clean_df['Abnormal_Return'].values

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # OLS estimation with error handling for singular matrix
        try:
            betas = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y
        except np.linalg.LinAlgError:
            return self._empty_result("Regression Analysis - Singular matrix")

        # Calculate statistics
        y_pred = X_with_intercept @ betas
        residuals = y - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Standard errors
        n = len(y)
        k = len(betas)
        mse = ss_res / (n - k)
        var_beta = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        se_beta = np.sqrt(np.diag(var_beta))

        # t-statistics and p-values
        t_stats = betas / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

        return {
            'Test': 'OLS Regression',
            'N': n,
            'Intercept': betas[0],
            'Intercept_SE': se_beta[0],
            'Intercept_t': t_stats[0],
            'Intercept_p': p_values[0],
            'News_Coefficient': betas[1],
            'News_SE': se_beta[1],
            'News_t': t_stats[1],
            'News_p': p_values[1],
            'R_squared': r_squared,
            'MSE': mse,
            'Significant': p_values[1] < self.alpha,
            'Interpretation': f"News days have {betas[1]:.4f} higher AR on average"
        }

    def test_news_impact(self, ar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive test suite for news impact

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns and News_Day indicator

        Returns:
        --------
        pd.DataFrame with all test results
        """
        results = []

        # Separate news and non-news days
        news_ar = ar_df[ar_df['News_Day'] == True]['Abnormal_Return']
        non_news_ar = ar_df[ar_df['News_Day'] == False]['Abnormal_Return']

        # Test 1: One-sample t-test on news days
        results.append(self.one_sample_ttest(news_ar, null_mean=0.0))

        # Test 2: One-sample t-test on non-news days
        results.append(self.one_sample_ttest(non_news_ar, null_mean=0.0))

        # Test 3: Two-sample t-test
        results.append(self.two_sample_ttest(news_ar, non_news_ar))

        # Test 4: Variance test
        results.append(self.variance_test(news_ar, non_news_ar))

        # Test 5: Regression analysis
        results.append(self.regression_analysis(ar_df))

        return pd.DataFrame(results)

    def subperiod_analysis(self,
                          ar_df: pd.DataFrame,
                          n_periods: int = 3) -> pd.DataFrame:
        """
        Robustness check: Test across different time periods

        Parameters:
        -----------
        ar_df : pd.DataFrame
            DataFrame with abnormal returns
        n_periods : int
            Number of subperiods to split data into

        Returns:
        --------
        pd.DataFrame with results for each subperiod
        """
        results = []
        total_days = len(ar_df)
        period_size = total_days // n_periods

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else total_days

            subperiod = ar_df.iloc[start_idx:end_idx]
            news_ar = subperiod[subperiod['News_Day'] == True]['Abnormal_Return']

            if len(news_ar) > 0:
                test_result = self.one_sample_ttest(news_ar)
                test_result['Period'] = f"Period {i+1}"
                test_result['Start_Date'] = subperiod.index[0]
                test_result['End_Date'] = subperiod.index[-1]
                results.append(test_result)

        return pd.DataFrame(results)

    def _empty_result(self, test_name: str) -> Dict:
        """Return empty result for insufficient data"""
        return {
            'Test': test_name,
            'Error': 'Insufficient data for test',
            'Significant': False,
            'p_value': np.nan
        }


def run_statistical_tests(ar_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Convenience function to run all statistical tests

    Returns:
    --------
    pd.DataFrame with test results
    """
    tester = StatisticalTester(alpha=alpha)
    return tester.test_news_impact(ar_df)


if __name__ == "__main__":
    print("Statistical Testing Module")
    print("Tests whether news significantly impacts stock prices")
    print("Includes t-tests, F-tests, regression, and robustness checks")
