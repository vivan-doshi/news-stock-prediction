"""
Experimental Analysis Pipeline - Version 2
Extended event window (-5 to +1) with relaxed polarity threshold and sentiment stratification

EXPERIMENT RATIONALE:
====================
Based on initial findings (no statistical significance), this experiment tests three hypotheses:

1. EVENT WINDOW HYPOTHESIS:
   - Original: Same-day (t=0) measurement
   - Problem: News published during trading day is already priced in by close
   - New approach: [-5, +1] window captures pre-announcement effects and next-day reaction

2. FILTER SENSITIVITY HYPOTHESIS:
   - Original: |polarity| > 0.5 (very strong sentiment only)
   - Problem: Too restrictive - only 33 AAPL events from 25,275 articles (0.13%)
   - New approach: |polarity| > 0.3 (moderate-to-strong sentiment)
   - Expected: 2-3x more events → better statistical power

3. STRATIFICATION HYPOTHESIS:
   - Original: Binary (news day vs non-news day)
   - Problem: Ignores sentiment strength variation
   - New approach: Stratify by sentiment quartiles
   - Expected: Non-linear relationship between sentiment strength and returns

EXPERIMENTAL PARAMETERS:
========================
- Event Window: [-5, +1] days (7-day window)
- Sentiment Threshold: |polarity| > 0.3 (vs 0.5 baseline)
- Stratification: 4 sentiment quartiles (very negative, negative, positive, very positive)
- All other parameters unchanged (Fama-French 5-factor, rolling 126-day estimation)

COMPARISON WITH BASELINE:
==========================
- Baseline results saved in: 03-output/results/{TICKER}/main_analysis/
- Experimental results saved in: 03-output/results/{TICKER}/experimental_v2/
- Side-by-side comparison report generated automatically
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
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


class ExperimentalAnalysisV2:
    """
    Experimental event study with:
    1. Extended event window (-5 to +1)
    2. Relaxed polarity threshold (0.3 vs 0.5)
    3. Sentiment stratification analysis
    """

    def __init__(self,
                 stock_file: str,
                 news_file: str,
                 ff_file: str,
                 ticker: str,
                 sector_file: Optional[str] = None,
                 data_dir: str = "../01-data",
                 output_dir: str = "../03-output"):
        """
        Parameters:
        -----------
        stock_file : str
            Path to stock price data
        news_file : str
            Path to news dates (raw news with sentiment scores)
        ff_file : str
            Path to Fama-French factors
        ticker : str
            Stock ticker symbol (for output organization)
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
        self.ticker = ticker
        self.sector_file = sector_file
        self.data_dir = data_dir

        # Create experimental output directory
        self.output_dir = Path(output_dir) / "results" / ticker / "experimental_v2"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Experimental parameters
        self.EVENT_WINDOW = (-5, 1)  # Extended window
        self.POLARITY_THRESHOLD = 0.3  # Relaxed threshold
        self.SENTIMENT_QUARTILES = 4  # For stratification

        # Data containers
        self.data = None
        self.news_dates = None
        self.news_df = None  # Full news data with sentiment
        self.beta_df = None
        self.ar_df = None
        self.test_results = None
        self.stratified_results = {}

        # Configuration
        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        if sector_file:
            self.factor_cols.append('Sector_Return')

    def run_complete_analysis(self) -> Dict:
        """
        Run the complete experimental analysis

        Returns:
        --------
        Dict with summary results
        """
        print("=" * 80)
        print("EXPERIMENTAL ANALYSIS V2: EXTENDED WINDOW + STRATIFICATION")
        print("=" * 80)

        # Document experimental setup
        self._document_experiment()

        # Step 1: Load and filter data with new parameters
        print("\n[Step 1/7] Loading and filtering data (polarity > 0.3)...")
        self._load_and_filter_data()

        # Step 2: Estimate betas (exclude extended event window)
        print("\n[Step 2/7] Estimating factor betas (excluding [-5,+1] window)...")
        self._estimate_betas()

        # Step 3: Calculate abnormal returns
        print("\n[Step 3/7] Calculating abnormal returns...")
        self._calculate_abnormal_returns()

        # Step 4: Tag news days with extended window
        print("\n[Step 4/7] Tagging news days (window = [-5, +1])...")
        self._tag_news_days_extended()

        # Step 5: Stratification analysis
        print("\n[Step 5/7] Running sentiment stratification analysis...")
        self._stratification_analysis()

        # Step 6: Statistical tests
        print("\n[Step 6/7] Running statistical tests...")
        self._run_statistical_tests()

        # Step 7: Generate visualizations and report
        print("\n[Step 7/7] Generating visualizations and report...")
        self._create_visualizations()
        summary = self._generate_report()

        print("\n" + "=" * 80)
        print("EXPERIMENTAL ANALYSIS COMPLETE!")
        print("=" * 80)

        return summary

    def _document_experiment(self):
        """Document experimental parameters"""
        doc = f"""
EXPERIMENTAL DESIGN DOCUMENT
=============================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {self.ticker}

EXPERIMENTAL PARAMETERS:
- Event Window: {self.EVENT_WINDOW} days (baseline: (0, 0))
- Polarity Threshold: |polarity| > {self.POLARITY_THRESHOLD} (baseline: 0.5)
- Stratification: {self.SENTIMENT_QUARTILES} sentiment quartiles

RATIONALE:
1. Extended window captures pre-announcement effects and next-day reaction
2. Relaxed threshold increases sample size for better statistical power
3. Stratification tests non-linear sentiment-return relationship

HYPOTHESIS:
- H0: News sentiment has no impact on abnormal returns
- H1: Stronger sentiment → larger abnormal returns (monotonic relationship)
"""

        with open(self.output_dir / "EXPERIMENT_DESIGN.txt", "w") as f:
            f.write(doc)

        print(doc)

    def _load_and_filter_data(self):
        """Load data and apply relaxed polarity filter"""
        # Load raw news data
        news_path = Path(self.data_dir) / self.news_file
        self.news_df = pd.read_csv(news_path)
        self.news_df['date'] = pd.to_datetime(self.news_df['date'])

        original_count = len(self.news_df)

        # Apply relaxed polarity filter
        self.news_df = self.news_df[
            abs(self.news_df['sentiment_polarity']) > self.POLARITY_THRESHOLD
        ].copy()

        # Take one event per day (highest absolute polarity)
        self.news_df['abs_polarity'] = abs(self.news_df['sentiment_polarity'])
        self.news_df = self.news_df.sort_values('abs_polarity', ascending=False)
        self.news_df = self.news_df.groupby(self.news_df['date'].dt.date).first().reset_index(drop=True)

        filtered_count = len(self.news_df)

        print(f"  ✓ Original articles: {original_count:,}")
        print(f"  ✓ After |polarity| > {self.POLARITY_THRESHOLD} filter: {filtered_count:,}")
        print(f"  ✓ Filter reduction: {(1 - filtered_count/original_count)*100:.2f}%")
        print(f"  ✓ Baseline comparison: {filtered_count / 33:.1f}x more events than baseline (AAPL)")

        # Extract news dates for event study (remove timezone to match stock data)
        self.news_dates = pd.to_datetime(self.news_df['date']).dt.tz_localize(None).dt.normalize()

        # Load other data (stock and factors only, we already have news)
        loader = DataLoader(data_dir=self.data_dir)

        # Load stock data
        stock_df = loader.load_stock_data(self.stock_file)

        # Load Fama-French factors
        ff_df = loader.load_fama_french_factors(self.ff_file)

        # Merge stock and factor data
        self.data = stock_df.join(ff_df, how='inner')

        # Remove NA values
        self.data = self.data.dropna()

        # Calculate excess return (needed for beta estimation)
        self.data['Excess_Return'] = self.data['Return'] - self.data['RF']

        print(f"  ✓ Loaded {len(self.data)} days of stock/factor data")

        # Filter news events to only those on trading days
        # Ensure both are timezone-naive and normalized
        trading_days_normalized = pd.DatetimeIndex(self.data.index).normalize()
        news_dates_normalized = pd.to_datetime(self.news_df['date']).dt.tz_localize(None).dt.normalize()

        # Create boolean mask for matching dates
        mask = news_dates_normalized.isin(trading_days_normalized)
        self.news_df = self.news_df[mask].reset_index(drop=True)

        # Store normalized, timezone-naive dates in the dataframe for later use
        self.news_df['date_normalized'] = pd.to_datetime(self.news_df['date']).dt.tz_localize(None).dt.normalize()
        self.news_dates = self.news_df['date_normalized']

        print(f"  ✓ Filtered to {len(self.news_df)} news events on trading days")

    def _estimate_betas(self):
        """Estimate factor betas excluding extended event window"""
        estimator = BetaEstimator(window_size=126, min_periods=100)
        self.beta_df = estimator.rolling_beta_estimation(
            data=self.data,
            factor_cols=self.factor_cols,
            exclude_dates=self.news_dates,
            event_window=self.EVENT_WINDOW  # Exclude [-5, +1]
        )

        print(f"  ✓ Estimated betas for {len(self.beta_df)} days")
        print(f"  ✓ Average R²: {self.beta_df['R_squared'].mean():.3f}")

        # Save beta estimates
        self.beta_df.to_csv(self.output_dir / "beta_estimates.csv")

    def _calculate_abnormal_returns(self):
        """Calculate abnormal returns"""
        calculator = AbnormalReturnsCalculator()

        # Calculate AR
        self.ar_df = calculator.calculate_abnormal_returns(
            data=self.data,
            beta_df=self.beta_df,
            factor_cols=self.factor_cols
        )

        print(f"  ✓ Calculated AR for {len(self.ar_df)} days")

        # Save AR
        self.ar_df.to_csv(self.output_dir / "abnormal_returns.csv")

    def _tag_news_days_extended(self):
        """Tag news days using extended window"""
        self.ar_df['News_Day'] = False
        self.ar_df['Days_From_News'] = np.nan
        self.ar_df['News_Sentiment'] = np.nan

        tagged_count = 0
        for _, news_row in self.news_df.iterrows():
            news_date = news_row['date_normalized']  # Already normalized and tz-naive
            sentiment = news_row['sentiment_polarity']

            # Tag all days in [-5, +1] window
            for offset in range(self.EVENT_WINDOW[0], self.EVENT_WINDOW[1] + 1):
                target_date = news_date + pd.Timedelta(days=offset)

                # Find nearest trading day (handle weekends/holidays)
                # Check if exact date exists
                if target_date in self.ar_df.index:
                    self.ar_df.loc[target_date, 'News_Day'] = True
                    self.ar_df.loc[target_date, 'Days_From_News'] = offset
                    # Only tag sentiment for the actual event day (offset=0)
                    if offset == 0:
                        self.ar_df.loc[target_date, 'News_Sentiment'] = sentiment
                    tagged_count += 1

        news_count = self.ar_df['News_Day'].sum()
        print(f"  ✓ Tagged {news_count} trading days as news days")
        print(f"  ✓ Event coverage: {news_count / len(self.ar_df) * 100:.1f}% of total days")
        print(f"  ✓ Processed {len(self.news_df)} news events")

    def _stratification_analysis(self):
        """Stratify news days by sentiment quartiles"""
        news_data = self.ar_df[self.ar_df['News_Day'] == True].copy()

        # Filter to only days with sentiment (actual event days, not pre/post days)
        news_data = news_data[news_data['News_Sentiment'].notna()].copy()

        if len(news_data) < self.SENTIMENT_QUARTILES:
            print(f"  ⚠ Insufficient data for stratification ({len(news_data)} days)")
            self.stratified_results['quartile_stats'] = pd.DataFrame()
            self.stratified_results['sentiment_correlation'] = {'spearman_rho': np.nan, 'p_value': np.nan, 'significant': False}
            return

        # Create sentiment quartiles (let pandas handle labels with duplicates='drop')
        try:
            news_data['Sentiment_Quartile'] = pd.qcut(
                news_data['News_Sentiment'],
                q=self.SENTIMENT_QUARTILES,
                duplicates='drop'
            )
            # Get unique quartile labels that were created
            quartile_labels = news_data['Sentiment_Quartile'].unique().tolist()
            quartile_labels = sorted(quartile_labels)
            print(f"  ✓ Created {len(quartile_labels)} sentiment bins")
        except (ValueError, TypeError) as e:
            # If quartiles can't be created, use binary split
            median = news_data['News_Sentiment'].median()
            news_data['Sentiment_Quartile'] = news_data['News_Sentiment'].apply(
                lambda x: 'Negative' if x < median else 'Positive'
            )
            quartile_labels = ['Negative', 'Positive']
            print(f"  ⚠ Using binary split instead of quartiles: {str(e)}")

        # Calculate statistics by quartile
        quartile_stats = []
        for quartile in quartile_labels:
            q_data = news_data[news_data['Sentiment_Quartile'] == quartile]['Abnormal_Return']

            quartile_stats.append({
                'Quartile': quartile,
                'N': len(q_data),
                'Mean_AR': q_data.mean(),
                'Median_AR': q_data.median(),
                'Std_AR': q_data.std(),
                'Min_AR': q_data.min(),
                'Max_AR': q_data.max(),
                'Mean_Sentiment': news_data[news_data['Sentiment_Quartile'] == quartile]['News_Sentiment'].mean()
            })

        self.stratified_results['quartile_stats'] = pd.DataFrame(quartile_stats)

        # Test monotonic relationship (Spearman correlation)
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(news_data['News_Sentiment'], news_data['Abnormal_Return'])

        self.stratified_results['sentiment_correlation'] = {
            'spearman_rho': corr,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

        print(f"  ✓ Stratified into {self.SENTIMENT_QUARTILES} quartiles")
        print(f"  ✓ Sentiment-Return Correlation: ρ={corr:.3f}, p={p_value:.4f}")

        # Save stratification results
        self.stratified_results['quartile_stats'].to_csv(
            self.output_dir / "sentiment_stratification.csv",
            index=False
        )

    def _run_statistical_tests(self):
        """Run statistical significance tests"""
        tester = StatisticalTester(alpha=0.05)

        # Main tests
        self.test_results = tester.test_news_impact(self.ar_df)

        # Additional test: Compare each quartile to non-news days
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return']

        quartile_tests = []
        for _, row in self.stratified_results['quartile_stats'].iterrows():
            quartile_name = row['Quartile']
            news_data = self.ar_df[self.ar_df['News_Day'] == True]

            # Get this quartile's data (need to recreate quartile assignment)
            news_data_copy = news_data.copy()
            try:
                news_data_copy['Sentiment_Quartile'] = pd.qcut(
                    news_data_copy['News_Sentiment'],
                    q=self.SENTIMENT_QUARTILES,
                    duplicates='drop'
                )
            except (ValueError, TypeError):
                # If quartiles can't be created, use binary split
                median = news_data_copy['News_Sentiment'].median()
                news_data_copy['Sentiment_Quartile'] = news_data_copy['News_Sentiment'].apply(
                    lambda x: 'Negative' if x < median else 'Positive'
                )

            q_ar = news_data_copy[news_data_copy['Sentiment_Quartile'] == quartile_name]['Abnormal_Return']

            # T-test vs non-news days
            test_result = tester.two_sample_ttest(q_ar, non_news_ar)
            test_result['Quartile'] = quartile_name
            test_result['Test'] = f'{quartile_name} vs Non-News'
            quartile_tests.append(test_result)

        self.stratified_results['quartile_tests'] = pd.DataFrame(quartile_tests)

        print(f"  ✓ Completed {len(self.test_results)} main statistical tests")
        print(f"  ✓ Completed {len(quartile_tests)} quartile comparison tests")

        # Save results
        self.test_results.to_csv(self.output_dir / "statistical_tests.csv", index=False)
        self.stratified_results['quartile_tests'].to_csv(
            self.output_dir / "quartile_tests.csv",
            index=False
        )

    def _create_visualizations(self):
        """Create experimental visualizations"""
        sns.set_style("whitegrid")

        # Figure 1: Extended window effect
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1.1: AR by days from news event
        news_data = self.ar_df[self.ar_df['News_Day'] == True].copy()
        window_stats = news_data.groupby('Days_From_News')['Abnormal_Return'].agg(['mean', 'std', 'count'])

        axes[0, 0].bar(window_stats.index, window_stats['mean'], alpha=0.7, color='steelblue')
        axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 0].set_xlabel('Days from News Event')
        axes[0, 0].set_ylabel('Mean Abnormal Return')
        axes[0, 0].set_title('AR by Event Window Position [-5 to +1]')
        axes[0, 0].grid(alpha=0.3)

        # 1.2: Sentiment quartiles comparison
        quartile_stats = self.stratified_results['quartile_stats']
        x_pos = np.arange(len(quartile_stats))
        axes[0, 1].bar(x_pos, quartile_stats['Mean_AR'], alpha=0.7, color='coral')
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(quartile_stats['Quartile'], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Mean Abnormal Return')
        axes[0, 1].set_title('AR by Sentiment Quartile')
        axes[0, 1].grid(alpha=0.3)

        # 1.3: Sentiment vs AR scatter
        axes[1, 0].scatter(news_data['News_Sentiment'], news_data['Abnormal_Return'],
                          alpha=0.5, s=20, color='green')
        axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=0.5)
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=0.5)

        # Add regression line
        z = np.polyfit(news_data['News_Sentiment'].dropna(),
                      news_data.loc[news_data['News_Sentiment'].notna(), 'Abnormal_Return'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(news_data['News_Sentiment'].min(), news_data['News_Sentiment'].max(), 100)
        axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')

        corr_info = self.stratified_results['sentiment_correlation']
        axes[1, 0].set_xlabel('News Sentiment Polarity')
        axes[1, 0].set_ylabel('Abnormal Return')
        axes[1, 0].set_title(f'Sentiment vs AR (ρ={corr_info["spearman_rho"]:.3f}, p={corr_info["p_value"]:.4f})')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 1.4: Sample size comparison
        comparison_data = {
            'Baseline\n(|pol|>0.5)': 33,  # AAPL baseline
            'Experimental\n(|pol|>0.3)': len(self.news_df),
            'Full Dataset': len(self.ar_df)
        }
        axes[1, 1].bar(comparison_data.keys(), comparison_data.values(),
                      alpha=0.7, color=['red', 'green', 'gray'])
        axes[1, 1].set_ylabel('Number of Events/Days')
        axes[1, 1].set_title('Sample Size Comparison')
        axes[1, 1].set_yscale('log')
        for i, (k, v) in enumerate(comparison_data.items()):
            axes[1, 1].text(i, v, f'{v}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(self.output_dir / "experimental_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved experimental visualizations")

    def _generate_report(self) -> Dict:
        """Generate experimental summary report"""
        news_ar = self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return']
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return']

        summary = {
            'ticker': self.ticker,
            'event_window': f"{self.EVENT_WINDOW}",
            'polarity_threshold': self.POLARITY_THRESHOLD,
            'total_days': len(self.ar_df),
            'news_events': len(self.news_df),
            'news_days': len(news_ar),
            'non_news_days': len(non_news_ar),
            'mean_ar_news': news_ar.mean(),
            'mean_ar_non_news': non_news_ar.mean(),
            'ar_difference': news_ar.mean() - non_news_ar.mean(),
            'std_ar_news': news_ar.std(),
            'std_ar_non_news': non_news_ar.std(),
            'avg_r_squared': self.beta_df['R_squared'].mean(),
            'sentiment_correlation': self.stratified_results['sentiment_correlation']['spearman_rho'],
            'sentiment_correlation_p': self.stratified_results['sentiment_correlation']['p_value'],
            'significant_tests': self.test_results['Significant'].sum() if 'Significant' in self.test_results.columns else 0,
            'total_tests': len(self.test_results)
        }

        # Print report
        print("\n" + "=" * 80)
        print("EXPERIMENTAL SUMMARY REPORT")
        print("=" * 80)
        print(f"\nExperimental Design:")
        print(f"  Event Window: {self.EVENT_WINDOW} days")
        print(f"  Polarity Threshold: |polarity| > {self.POLARITY_THRESHOLD}")
        print(f"  Sentiment Stratification: {self.SENTIMENT_QUARTILES} quartiles")

        print(f"\nSample Size:")
        print(f"  News events: {summary['news_events']} (vs 33 baseline)")
        print(f"  News days: {summary['news_days']} ({summary['news_days']/summary['total_days']*100:.1f}% coverage)")
        print(f"  Sample size increase: {summary['news_events']/33:.1f}x")

        print(f"\nAbnormal Returns:")
        print(f"  Mean AR (News Days): {summary['mean_ar_news']:.4f} ({summary['mean_ar_news']*100:.2f}%)")
        print(f"  Mean AR (Non-News Days): {summary['mean_ar_non_news']:.4f} ({summary['mean_ar_non_news']*100:.2f}%)")
        print(f"  Difference: {summary['ar_difference']:.4f} ({summary['ar_difference']*100:.2f}%)")

        print(f"\nSentiment Analysis:")
        print(f"  Sentiment-Return Correlation: ρ={summary['sentiment_correlation']:.3f}")
        print(f"  Correlation p-value: {summary['sentiment_correlation_p']:.4f}")
        print(f"  Significant correlation: {'YES' if summary['sentiment_correlation_p'] < 0.05 else 'NO'}")

        print(f"\nStatistical Tests:")
        print(f"  Significant tests: {summary['significant_tests']} / {summary['total_tests']}")
        print(f"  Model fit (R²): {summary['avg_r_squared']:.3f}")

        # Comparison with baseline
        print(f"\n{'-' * 80}")
        print("COMPARISON WITH BASELINE:")
        print(f"{'-' * 80}")
        print(f"  Sample size: {summary['news_events']/33:.1f}x larger")
        print(f"  Event window: 7 days vs 1 day (baseline)")
        print(f"  Effect size: {abs(summary['ar_difference'])/abs(0.001067):.1f}x baseline")

        # Save summary
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(self.output_dir / "experimental_summary.csv", index=False)

        return summary


def main():
    """
    Run experimental analysis for AAPL and TSLA
    """
    tickers = ['AAPL', 'TSLA']

    for ticker in tickers:
        print(f"\n{'=' * 80}")
        print(f"STARTING EXPERIMENTAL ANALYSIS FOR {ticker}")
        print(f"{'=' * 80}\n")

        analysis = ExperimentalAnalysisV2(
            stock_file=f"{ticker}_stock_data.csv",
            news_file=f"{ticker}_eodhd_news.csv",
            ff_file="fama_french_factors.csv",
            ticker=ticker,
            sector_file=None,
            data_dir="../01-data",
            output_dir="../03-output"
        )

        try:
            summary = analysis.run_complete_analysis()
            print(f"\n✓ {ticker} experimental analysis completed successfully!")
        except Exception as e:
            print(f"\n✗ {ticker} experimental analysis failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
