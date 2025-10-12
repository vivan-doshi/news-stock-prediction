"""
Main Analysis Pipeline for Phase 1: News Impact Detection
Orchestrates the complete event study analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules using importlib to handle numeric prefixes
import importlib
data_loader_module = importlib.import_module('01_data_loader')
beta_estimation_module = importlib.import_module('02_beta_estimation')
abnormal_returns_module = importlib.import_module('03_abnormal_returns')
statistical_tests_module = importlib.import_module('04_statistical_tests')

DataLoader = data_loader_module.DataLoader
load_all_data = data_loader_module.load_all_data
BetaEstimator = beta_estimation_module.BetaEstimator
estimate_betas = beta_estimation_module.estimate_betas
AbnormalReturnsCalculator = abnormal_returns_module.AbnormalReturnsCalculator
calculate_abnormal_returns = abnormal_returns_module.calculate_abnormal_returns
StatisticalTester = statistical_tests_module.StatisticalTester
run_statistical_tests = statistical_tests_module.run_statistical_tests


class Phase1Analysis:
    """Complete Phase 1 analysis pipeline"""

    def __init__(self,
                 stock_file: str,
                 news_file: str,
                 ff_file: str,
                 sector_file: Optional[str] = None,
                 data_dir: str = "../01-data",
                 output_dir: str = "../03-output"):
        """
        Parameters:
        -----------
        stock_file : str
            Path to stock price data
        news_file : str
            Path to news dates
        ff_file : str
            Path to Fama-French factors
        sector_file : str, optional
            Path to sector factors
        data_dir : str
            Data directory path
        output_dir : str
            Output directory for results
        """
        self.stock_file = stock_file
        self.news_file = news_file
        self.ff_file = ff_file
        self.sector_file = sector_file
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Data containers
        self.data = None
        self.news_dates = None
        self.beta_df = None
        self.ar_df = None
        self.test_results = None

        # Configuration
        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        if sector_file:
            self.factor_cols.append('Sector_Return')

    def run_complete_analysis(self) -> Dict:
        """
        Run the complete Phase 1 analysis

        Returns:
        --------
        Dict with summary results
        """
        print("=" * 70)
        print("PHASE 1: NEWS IMPACT DETECTION - EVENT STUDY ANALYSIS")
        print("=" * 70)

        # Step 1: Load data
        print("\n[Step 1/6] Loading data...")
        self._load_data()

        # Step 2: Estimate betas
        print("\n[Step 2/6] Estimating factor betas...")
        self._estimate_betas()

        # Step 3: Calculate abnormal returns
        print("\n[Step 3/6] Calculating abnormal returns...")
        self._calculate_abnormal_returns()

        # Step 4: Statistical tests
        print("\n[Step 4/6] Running statistical tests...")
        self._run_statistical_tests()

        # Step 5: Generate visualizations
        print("\n[Step 5/6] Generating visualizations...")
        self._create_visualizations()

        # Step 6: Generate report
        print("\n[Step 6/6] Generating summary report...")
        summary = self._generate_report()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE!")
        print("=" * 70)

        return summary

    def _load_data(self):
        """Load all required data"""
        self.data, self.news_dates = load_all_data(
            stock_file=self.stock_file,
            news_file=self.news_file,
            ff_file=self.ff_file,
            sector_file=self.sector_file,
            data_dir=self.data_dir
        )
        print(f"  ✓ Loaded {len(self.data)} days of data")
        print(f"  ✓ Loaded {len(self.news_dates)} news events")

    def _estimate_betas(self):
        """Estimate factor betas using rolling windows"""
        estimator = BetaEstimator(window_size=126, min_periods=100)
        self.beta_df = estimator.rolling_beta_estimation(
            data=self.data,
            factor_cols=self.factor_cols,
            exclude_dates=self.news_dates,
            event_window=(-1, 2)
        )

        # Calculate beta stability
        stability = estimator.calculate_beta_stability(self.beta_df, self.factor_cols)
        print(f"  ✓ Estimated betas for {len(self.beta_df)} days")
        print(f"  ✓ Average R²: {self.beta_df['R_squared'].mean():.3f}")

        # Save beta estimates
        self.beta_df.to_csv(self.output_dir / "beta_estimates.csv")
        stability.to_csv(self.output_dir / "beta_stability.csv")

    def _calculate_abnormal_returns(self):
        """Calculate abnormal returns"""
        calculator = AbnormalReturnsCalculator()

        # Calculate AR
        self.ar_df = calculator.calculate_abnormal_returns(
            data=self.data,
            beta_df=self.beta_df,
            factor_cols=self.factor_cols
        )

        # Tag news days
        self.ar_df = calculator.tag_news_days(
            ar_df=self.ar_df,
            news_dates=self.news_dates,
            window=(0, 0)
        )

        # Calculate CAR
        car_df = calculator.calculate_cumulative_abnormal_returns(
            ar_df=self.ar_df,
            news_dates=self.news_dates,
            window=(-1, 2)
        )

        # Summary statistics
        stats = calculator.calculate_ar_statistics(self.ar_df)

        print(f"  ✓ Calculated AR for {len(self.ar_df)} days")

        # Find news and non-news stats rows
        news_stats = stats[stats['Category'] == 'News Days']
        non_news_stats = stats[stats['Category'] == 'Non-News Days']

        if len(news_stats) > 0:
            print(f"  ✓ Mean AR (News Days): {news_stats.iloc[0]['Mean']:.4f}")
        if len(non_news_stats) > 0:
            print(f"  ✓ Mean AR (Non-News Days): {non_news_stats.iloc[0]['Mean']:.4f}")

        # Save results
        self.ar_df.to_csv(self.output_dir / "abnormal_returns.csv")
        car_df.to_csv(self.output_dir / "cumulative_abnormal_returns.csv", index=False)
        stats.to_csv(self.output_dir / "ar_statistics.csv", index=False)

    def _run_statistical_tests(self):
        """Run statistical significance tests"""
        tester = StatisticalTester(alpha=0.05)

        # Main tests
        self.test_results = tester.test_news_impact(self.ar_df)

        # Subperiod analysis for robustness
        subperiod_results = tester.subperiod_analysis(self.ar_df, n_periods=3)

        print(f"  ✓ Completed {len(self.test_results)} statistical tests")

        # Print key results
        for _, test in self.test_results.iterrows():
            if 'p_value' in test:
                sig = "✓ Significant" if test['Significant'] else "✗ Not significant"
                print(f"  {test['Test']}: p={test['p_value']:.4f} {sig}")

        # Save results
        self.test_results.to_csv(self.output_dir / "statistical_tests.csv", index=False)
        subperiod_results.to_csv(self.output_dir / "subperiod_analysis.csv", index=False)

    def _create_visualizations(self):
        """Create visualizations"""
        sns.set_style("whitegrid")

        # 1. AR Distribution by News/Non-News Days
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Histogram
        news_ar = self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return']
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return']

        axes[0, 0].hist(news_ar.dropna(), bins=30, alpha=0.7, label='News Days', color='red')
        axes[0, 0].hist(non_news_ar.dropna(), bins=30, alpha=0.7, label='Non-News Days', color='blue')
        axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('Abnormal Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Abnormal Returns')
        axes[0, 0].legend()

        # Box plot
        self.ar_df.boxplot(column='Abnormal_Return', by='News_Day', ax=axes[0, 1])
        axes[0, 1].set_xlabel('News Day')
        axes[0, 1].set_ylabel('Abnormal Return')
        axes[0, 1].set_title('AR by News Day')
        plt.sca(axes[0, 1])
        plt.xticks([1, 2], ['False', 'True'])

        # Time series
        axes[1, 0].plot(self.ar_df.index, self.ar_df['Abnormal_Return'], alpha=0.5, linewidth=0.5)
        axes[1, 0].scatter(self.ar_df[self.ar_df['News_Day'] == True].index,
                          self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return'],
                          color='red', s=20, label='News Days', alpha=0.7)
        axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Abnormal Return')
        axes[1, 0].set_title('Abnormal Returns Over Time')
        axes[1, 0].legend()

        # R-squared over time
        axes[1, 1].plot(self.beta_df.index, self.beta_df['R_squared'], linewidth=1)
        axes[1, 1].axhline(self.beta_df['R_squared'].mean(), color='red',
                          linestyle='--', label=f"Mean: {self.beta_df['R_squared'].mean():.3f}")
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('R²')
        axes[1, 1].set_title('Model Fit (R²) Over Time')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "analysis_summary.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualizations to {self.output_dir}")

    def _generate_report(self) -> Dict:
        """Generate summary report"""
        news_ar = self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return']
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return']

        # Safely count significant tests
        significant_tests = 0
        if 'Significant' in self.test_results.columns:
            significant_tests = self.test_results['Significant'].sum()

        summary = {
            'total_days': len(self.ar_df),
            'news_days': len(news_ar),
            'non_news_days': len(non_news_ar),
            'mean_ar_news': news_ar.mean(),
            'mean_ar_non_news': non_news_ar.mean(),
            'std_ar_news': news_ar.std(),
            'std_ar_non_news': non_news_ar.std(),
            'avg_r_squared': self.beta_df['R_squared'].mean(),
            'significant_tests': significant_tests,
            'total_tests': len(self.test_results)
        }

        # Print report
        print("\n" + "=" * 70)
        print("SUMMARY REPORT")
        print("=" * 70)
        print(f"\nData Overview:")
        print(f"  Total days analyzed: {summary['total_days']}")
        print(f"  News days: {summary['news_days']}")
        print(f"  Non-news days: {summary['non_news_days']}")
        print(f"\nAbnormal Returns:")
        print(f"  Mean AR (News Days): {summary['mean_ar_news']:.4f} ({summary['mean_ar_news']*100:.2f}%)")
        print(f"  Mean AR (Non-News Days): {summary['mean_ar_non_news']:.4f} ({summary['mean_ar_non_news']*100:.2f}%)")
        print(f"  Std AR (News Days): {summary['std_ar_news']:.4f}")
        print(f"  Std AR (Non-News Days): {summary['std_ar_non_news']:.4f}")
        print(f"\nModel Performance:")
        print(f"  Average R²: {summary['avg_r_squared']:.3f}")
        print(f"\nStatistical Significance:")
        print(f"  Significant tests: {summary['significant_tests']} / {summary['total_tests']}")

        # Phase 1 success criteria
        print(f"\n{'=' * 70}")
        print("PHASE 1 SUCCESS CRITERIA:")
        print("=" * 70)

        criteria_met = []

        # Criterion 1: Significant p-value
        sig_count = summary['significant_tests']
        if sig_count >= 3:
            print("  ✓ Multiple significant tests (p < 0.05)")
            criteria_met.append(True)
        else:
            print("  ✗ Insufficient significant tests")
            criteria_met.append(False)

        # Criterion 2: AR magnitude
        ar_pct = abs(summary['mean_ar_news']) * 100
        if 1.0 <= ar_pct <= 5.0:
            print(f"  ✓ Mean AR in expected range: {ar_pct:.2f}%")
            criteria_met.append(True)
        else:
            print(f"  ⚠ Mean AR outside expected range: {ar_pct:.2f}%")
            criteria_met.append(False)

        # Criterion 3: Model fit
        if summary['avg_r_squared'] >= 0.3:
            print(f"  ✓ Good model fit: R² = {summary['avg_r_squared']:.3f}")
            criteria_met.append(True)
        else:
            print(f"  ⚠ Low model fit: R² = {summary['avg_r_squared']:.3f}")
            criteria_met.append(False)

        # Overall verdict
        if all(criteria_met):
            print("\n✓ PHASE 1 SUCCESSFUL: News significantly impacts stock prices!")
        elif sum(criteria_met) >= 2:
            print("\n⚠ PHASE 1 PARTIALLY SUCCESSFUL: Some evidence of news impact")
        else:
            print("\n✗ PHASE 1 UNSUCCESSFUL: No clear evidence of news impact")

        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "analysis_summary.csv", index=False)

        return summary


def main():
    """Example usage"""
    # Configure file paths
    analysis = Phase1Analysis(
        stock_file="stock_prices.csv",
        news_file="news_dates.csv",
        ff_file="fama_french_factors.csv",
        sector_file=None,  # Optional
        data_dir="../01-data",
        output_dir="../03-output"
    )

    # Run complete analysis
    summary = analysis.run_complete_analysis()


if __name__ == "__main__":
    print("Phase 1 Analysis Pipeline")
    print("To run analysis, configure file paths and call run_complete_analysis()")
