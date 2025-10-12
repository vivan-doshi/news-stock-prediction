"""
Experimental Analysis Pipeline - Version 3 (OPTIMIZED)
Balanced approach with validation-driven parameter selection

CRITICAL IMPROVEMENTS FROM V2:
================================
V2 FAILURE: Event window too large (93.5% coverage) → no data for beta estimation → all NaN results

V3 SOLUTION:
1. OPTIMIZED EVENT WINDOW: [-1, +1] days (3-day window)
   - Captures pre-announcement leakage (t-1) and next-day reaction (t+1)
   - Sufficient non-event days remain for beta estimation
   - Standard in finance literature (see MacKinlay 1997)

2. CALIBRATED POLARITY THRESHOLD: |polarity| > 0.6 (vs 0.3 in V2, 0.5 baseline)
   - V2's 0.3 captured too many weak signals (95% were polarity=1.0)
   - 0.6 targets genuinely strong sentiment
   - Balances sample size increase with signal quality

3. SIMPLIFIED STRATIFICATION: 3 categories (Negative, Neutral, Positive)
   - More robust than quartiles with limited data
   - Based on sentiment terciles for equal groups
   - Reduces overfitting risk

EXPERIMENTAL PARAMETERS:
========================
- Event Window: [-1, +1] days (3-day window, ~10-15% coverage expected)
- Sentiment Threshold: |polarity| > 0.6 (strong sentiment only)
- Stratification: 3 sentiment categories (terciles)
- Beta Estimation: 126-day rolling window, 100-day minimum
- Statistical Test: Welch's t-test (unequal variances)

EXPECTED OUTCOMES:
==================
- Sample size: 3-5x baseline (100-150 events for AAPL)
- Event coverage: 10-20% of trading days (sufficient for beta estimation)
- Valid abnormal returns and statistical tests
- Interpretable sentiment stratification results

SCIENTIFIC RATIONALE:
====================
This design follows established event study methodology:
- Short event windows reduce noise (Fama 1991)
- Higher sentiment thresholds improve signal-to-noise ratio
- Tercile stratification is standard for subgroup analysis
- Preserves statistical power while ensuring valid estimation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import importlib
data_loader_module = importlib.import_module('01_data_loader')
beta_estimation_module = importlib.import_module('02_beta_estimation')
abnormal_returns_module = importlib.import_module('03_abnormal_returns')
statistical_tests_module = importlib.import_module('04_statistical_tests')

DataLoader = data_loader_module.DataLoader
BetaEstimator = beta_estimation_module.BetaEstimator
AbnormalReturnsCalculator = abnormal_returns_module.AbnormalReturnsCalculator
StatisticalTester = statistical_tests_module.StatisticalTester


class ExperimentalAnalysisV3:
    """
    Optimized experimental event study with:
    1. Balanced event window [-1, +1]
    2. Calibrated polarity threshold (0.6)
    3. Robust tercile stratification
    """

    def __init__(self,
                 stock_file: str,
                 news_file: str,
                 ff_file: str,
                 ticker: str,
                 data_dir: str = "../01-data",
                 output_dir: str = "../03-output",
                 event_window: Tuple[int, int] = (-1, 1),
                 polarity_threshold: float = 0.6):
        """Initialize experimental analysis with configurable parameters"""
        self.stock_file = stock_file
        self.news_file = news_file
        self.ff_file = ff_file
        self.ticker = ticker
        self.data_dir = data_dir

        # Experimental parameters (configurable)
        self.EVENT_WINDOW = event_window
        self.POLARITY_THRESHOLD = polarity_threshold
        self.SENTIMENT_CATEGORIES = 3  # Terciles

        # Create output directory
        self.output_dir = Path(output_dir) / "results" / ticker / "experimental_v3"
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Data containers
        self.data = None
        self.news_df = None
        self.news_dates = None
        self.beta_df = None
        self.ar_df = None
        self.test_results = None
        self.stratified_results = {}

        # Configuration
        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    def run_complete_analysis(self) -> Dict:
        """Run complete experimental analysis pipeline"""
        print("=" * 80)
        print(f"EXPERIMENTAL ANALYSIS V3: {self.ticker}")
        print("=" * 80)

        self._document_experiment()

        print("\n[Step 1/6] Loading and filtering data...")
        self._load_and_filter_data()

        print("\n[Step 2/6] Estimating factor betas...")
        self._estimate_betas()

        print("\n[Step 3/6] Calculating abnormal returns...")
        self._calculate_abnormal_returns()

        print("\n[Step 4/6] Tagging news days...")
        self._tag_news_days()

        print("\n[Step 5/6] Running stratification analysis...")
        self._stratification_analysis()

        print("\n[Step 6/6] Running statistical tests...")
        self._run_statistical_tests()

        print("\n[Visualization] Generating plots...")
        self._create_visualizations()

        print("\n[Report] Generating summary...")
        summary = self._generate_report()

        print("\n" + "=" * 80)
        print(f"✓ {self.ticker} ANALYSIS COMPLETE!")
        print("=" * 80)

        return summary

    def _document_experiment(self):
        """Document experimental design"""
        doc = f"""
EXPERIMENTAL DESIGN DOCUMENT - V3 (OPTIMIZED)
==============================================
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Ticker: {self.ticker}

EXPERIMENTAL PARAMETERS:
- Event Window: {self.EVENT_WINDOW} days
- Polarity Threshold: |polarity| > {self.POLARITY_THRESHOLD}
- Stratification: {self.SENTIMENT_CATEGORIES} sentiment categories (terciles)
- Beta Estimation: 126-day rolling window, 100-day minimum

IMPROVEMENTS OVER V2:
- Reduced event window to prevent data starvation
- Increased polarity threshold for stronger signals
- Simplified stratification for robustness

HYPOTHESIS:
H0: News sentiment has no impact on abnormal returns
H1: Stronger sentiment magnitude → larger absolute abnormal returns
"""
        with open(self.output_dir / "EXPERIMENT_DESIGN.txt", "w") as f:
            f.write(doc)
        print(doc)

    def _load_and_filter_data(self):
        """Load and filter data with optimized parameters"""
        # Load raw news data
        news_path = Path(self.data_dir) / self.news_file
        self.news_df = pd.read_csv(news_path)
        self.news_df['date'] = pd.to_datetime(self.news_df['date'])

        original_count = len(self.news_df)

        # Apply polarity filter
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
        print(f"  ✓ Retention rate: {filtered_count/original_count*100:.2f}%")

        # Load stock and factor data
        loader = DataLoader(data_dir=self.data_dir)
        stock_df = loader.load_stock_data(self.stock_file)
        ff_df = loader.load_fama_french_factors(self.ff_file)

        self.data = stock_df.join(ff_df, how='inner').dropna()
        self.data['Excess_Return'] = self.data['Return'] - self.data['RF']

        print(f"  ✓ Loaded {len(self.data)} days of stock/factor data")

        # Filter news to trading days only
        trading_days = pd.DatetimeIndex(self.data.index).normalize()
        news_dates = pd.to_datetime(self.news_df['date']).dt.tz_localize(None).dt.normalize()

        mask = news_dates.isin(trading_days)
        self.news_df = self.news_df[mask].reset_index(drop=True)
        self.news_df['date_normalized'] = pd.to_datetime(self.news_df['date']).dt.tz_localize(None).dt.normalize()
        self.news_dates = self.news_df['date_normalized']

        print(f"  ✓ {len(self.news_df)} news events on trading days")
        print(f"  ✓ Expected event coverage: ~{len(self.news_df) * (self.EVENT_WINDOW[1] - self.EVENT_WINDOW[0] + 1) / len(self.data) * 100:.1f}%")

    def _estimate_betas(self):
        """Estimate factor betas"""
        estimator = BetaEstimator(window_size=126, min_periods=100)
        self.beta_df = estimator.rolling_beta_estimation(
            data=self.data,
            factor_cols=self.factor_cols,
            exclude_dates=self.news_dates,
            event_window=self.EVENT_WINDOW
        )

        valid_betas = self.beta_df['R_squared'].notna().sum()
        avg_r2 = self.beta_df['R_squared'].mean()

        print(f"  ✓ Estimated betas for {len(self.beta_df)} days")
        print(f"  ✓ Valid beta estimates: {valid_betas} ({valid_betas/len(self.beta_df)*100:.1f}%)")
        print(f"  ✓ Average R²: {avg_r2:.3f}")

        if valid_betas < len(self.beta_df) * 0.5:
            print(f"  ⚠ WARNING: Only {valid_betas/len(self.beta_df)*100:.1f}% valid betas")

        self.beta_df.to_csv(self.output_dir / "beta_estimates.csv")

    def _calculate_abnormal_returns(self):
        """Calculate abnormal returns"""
        calculator = AbnormalReturnsCalculator()
        self.ar_df = calculator.calculate_abnormal_returns(
            data=self.data,
            beta_df=self.beta_df,
            factor_cols=self.factor_cols
        )

        valid_ar = self.ar_df['Abnormal_Return'].notna().sum()
        print(f"  ✓ Calculated AR for {len(self.ar_df)} days")
        print(f"  ✓ Valid AR: {valid_ar} ({valid_ar/len(self.ar_df)*100:.1f}%)")

        self.ar_df.to_csv(self.output_dir / "abnormal_returns.csv")

    def _tag_news_days(self):
        """Tag news days with event window"""
        self.ar_df['News_Day'] = False
        self.ar_df['Days_From_News'] = np.nan
        self.ar_df['News_Sentiment'] = np.nan

        for _, news_row in self.news_df.iterrows():
            news_date = news_row['date_normalized']
            sentiment = news_row['sentiment_polarity']

            for offset in range(self.EVENT_WINDOW[0], self.EVENT_WINDOW[1] + 1):
                target_date = news_date + pd.Timedelta(days=offset)

                if target_date in self.ar_df.index:
                    self.ar_df.loc[target_date, 'News_Day'] = True
                    self.ar_df.loc[target_date, 'Days_From_News'] = offset
                    if offset == 0:
                        self.ar_df.loc[target_date, 'News_Sentiment'] = sentiment

        news_count = self.ar_df['News_Day'].sum()
        coverage = news_count / len(self.ar_df) * 100

        print(f"  ✓ Tagged {news_count} trading days as news days")
        print(f"  ✓ Event coverage: {coverage:.1f}%")

        if coverage > 50:
            print(f"  ⚠ WARNING: High event coverage ({coverage:.1f}%) may limit beta estimation")

    def _stratification_analysis(self):
        """Stratify by sentiment terciles"""
        # Get event days with sentiment (t=0 only)
        news_data = self.ar_df[
            (self.ar_df['News_Day'] == True) &
            (self.ar_df['News_Sentiment'].notna())
        ].copy()

        if len(news_data) < self.SENTIMENT_CATEGORIES:
            print(f"  ⚠ Insufficient data for stratification ({len(news_data)} events)")
            self.stratified_results['tercile_stats'] = pd.DataFrame()
            self.stratified_results['sentiment_correlation'] = {
                'spearman_rho': np.nan, 'p_value': np.nan
            }
            return

        # Create terciles (negative, neutral, positive)
        try:
            news_data['Sentiment_Category'] = pd.qcut(
                news_data['News_Sentiment'],
                q=self.SENTIMENT_CATEGORIES,
                labels=['Negative', 'Neutral', 'Positive'],
                duplicates='drop'
            )
        except (ValueError, TypeError):
            # Fallback to binary split
            median = news_data['News_Sentiment'].median()
            news_data['Sentiment_Category'] = news_data['News_Sentiment'].apply(
                lambda x: 'Negative' if x < median else 'Positive'
            )

        # Calculate statistics by category
        categories = news_data['Sentiment_Category'].unique()
        tercile_stats = []

        for cat in categories:
            cat_data = news_data[news_data['Sentiment_Category'] == cat]
            tercile_stats.append({
                'Category': cat,
                'N': len(cat_data),
                'Mean_AR': cat_data['Abnormal_Return'].mean(),
                'Median_AR': cat_data['Abnormal_Return'].median(),
                'Std_AR': cat_data['Abnormal_Return'].std(),
                'Mean_Sentiment': cat_data['News_Sentiment'].mean()
            })

        self.stratified_results['tercile_stats'] = pd.DataFrame(tercile_stats)

        # Spearman correlation
        valid_data = news_data[['News_Sentiment', 'Abnormal_Return']].dropna()
        if len(valid_data) > 2:
            corr, p_value = spearmanr(valid_data['News_Sentiment'], valid_data['Abnormal_Return'])
            self.stratified_results['sentiment_correlation'] = {
                'spearman_rho': corr,
                'p_value': p_value
            }
            print(f"  ✓ Stratified into {len(categories)} categories")
            print(f"  ✓ Sentiment-Return Correlation: ρ={corr:.3f}, p={p_value:.4f}")
        else:
            self.stratified_results['sentiment_correlation'] = {
                'spearman_rho': np.nan,
                'p_value': np.nan
            }

        self.stratified_results['tercile_stats'].to_csv(
            self.output_dir / "sentiment_stratification.csv",
            index=False
        )

    def _run_statistical_tests(self):
        """Run statistical tests"""
        tester = StatisticalTester(alpha=0.05)

        # Main test: news vs non-news
        self.test_results = tester.test_news_impact(self.ar_df)

        print(f"  ✓ Completed {len(self.test_results)} statistical tests")

        # Count significant results
        if 'Significant' in self.test_results.columns:
            sig_count = self.test_results['Significant'].sum()
            print(f"  ✓ Significant results: {sig_count}/{len(self.test_results)}")

        self.test_results.to_csv(self.output_dir / "statistical_tests.csv", index=False)

    def _create_visualizations(self):
        """Generate visualizations"""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. AR by event window position
        news_data = self.ar_df[self.ar_df['News_Day'] == True].copy()
        if len(news_data) > 0 and 'Days_From_News' in news_data.columns:
            window_stats = news_data.groupby('Days_From_News')['Abnormal_Return'].agg(['mean', 'std', 'count'])
            axes[0, 0].bar(window_stats.index, window_stats['mean'], alpha=0.7, color='steelblue')
            axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 0].set_xlabel('Days from News Event')
            axes[0, 0].set_ylabel('Mean Abnormal Return')
            axes[0, 0].set_title(f'AR by Event Window Position {self.EVENT_WINDOW}')
            axes[0, 0].grid(alpha=0.3)

        # 2. Sentiment categories comparison
        tercile_stats = self.stratified_results.get('tercile_stats', pd.DataFrame())
        if not tercile_stats.empty:
            x_pos = np.arange(len(tercile_stats))
            axes[0, 1].bar(x_pos, tercile_stats['Mean_AR'], alpha=0.7, color='coral')
            axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(tercile_stats['Category'], rotation=0)
            axes[0, 1].set_ylabel('Mean Abnormal Return')
            axes[0, 1].set_title('AR by Sentiment Category')
            axes[0, 1].grid(alpha=0.3)

        # 3. Sentiment vs AR scatter
        plot_data = news_data[['News_Sentiment', 'Abnormal_Return']].dropna()
        if len(plot_data) > 0:
            axes[1, 0].scatter(plot_data['News_Sentiment'], plot_data['Abnormal_Return'],
                              alpha=0.5, s=30, color='green')
            axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=0.5)
            axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=0.5)

            # Regression line
            if len(plot_data) > 2:
                z = np.polyfit(plot_data['News_Sentiment'], plot_data['Abnormal_Return'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(plot_data['News_Sentiment'].min(),
                                    plot_data['News_Sentiment'].max(), 100)
                axes[1, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            corr_info = self.stratified_results.get('sentiment_correlation', {})
            rho = corr_info.get('spearman_rho', np.nan)
            p_val = corr_info.get('p_value', np.nan)
            axes[1, 0].set_xlabel('News Sentiment Polarity')
            axes[1, 0].set_ylabel('Abnormal Return')
            axes[1, 0].set_title(f'Sentiment vs AR (ρ={rho:.3f}, p={p_val:.4f})')
            axes[1, 0].grid(alpha=0.3)

        # 4. Distribution comparison
        news_ar = self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return'].dropna()
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return'].dropna()

        if len(news_ar) > 0 and len(non_news_ar) > 0:
            axes[1, 1].hist(news_ar, bins=30, alpha=0.6, label='News Days', color='blue', density=True)
            axes[1, 1].hist(non_news_ar, bins=30, alpha=0.6, label='Non-News Days', color='gray', density=True)
            axes[1, 1].axvline(news_ar.mean(), color='blue', linestyle='--', linewidth=2, label=f'News Mean: {news_ar.mean():.4f}')
            axes[1, 1].axvline(non_news_ar.mean(), color='gray', linestyle='--', linewidth=2, label=f'Non-News Mean: {non_news_ar.mean():.4f}')
            axes[1, 1].set_xlabel('Abnormal Return')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('AR Distribution: News vs Non-News Days')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "experimental_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualizations")

    def _generate_report(self) -> Dict:
        """Generate summary report"""
        news_ar = self.ar_df[self.ar_df['News_Day'] == True]['Abnormal_Return'].dropna()
        non_news_ar = self.ar_df[self.ar_df['News_Day'] == False]['Abnormal_Return'].dropna()

        summary = {
            'ticker': self.ticker,
            'event_window': str(self.EVENT_WINDOW),
            'polarity_threshold': self.POLARITY_THRESHOLD,
            'total_days': len(self.ar_df),
            'news_events': len(self.news_df),
            'news_days_tagged': len(news_ar),
            'non_news_days': len(non_news_ar),
            'event_coverage_pct': len(news_ar) / len(self.ar_df) * 100,
            'mean_ar_news': news_ar.mean() if len(news_ar) > 0 else np.nan,
            'mean_ar_non_news': non_news_ar.mean() if len(non_news_ar) > 0 else np.nan,
            'ar_difference': news_ar.mean() - non_news_ar.mean() if len(news_ar) > 0 and len(non_news_ar) > 0 else np.nan,
            'std_ar_news': news_ar.std() if len(news_ar) > 0 else np.nan,
            'std_ar_non_news': non_news_ar.std() if len(non_news_ar) > 0 else np.nan,
            'avg_r_squared': self.beta_df['R_squared'].mean(),
            'valid_betas_pct': self.beta_df['R_squared'].notna().sum() / len(self.beta_df) * 100,
            'sentiment_correlation': self.stratified_results.get('sentiment_correlation', {}).get('spearman_rho', np.nan),
            'sentiment_correlation_p': self.stratified_results.get('sentiment_correlation', {}).get('p_value', np.nan),
            'significant_tests': self.test_results['Significant'].sum() if 'Significant' in self.test_results.columns else 0,
            'total_tests': len(self.test_results)
        }

        # Print formatted report
        print("\n" + "=" * 80)
        print("EXPERIMENTAL SUMMARY REPORT")
        print("=" * 80)
        print(f"\nTicker: {summary['ticker']}")
        print(f"Event Window: {summary['event_window']} days")
        print(f"Polarity Threshold: |polarity| > {summary['polarity_threshold']}")

        print(f"\nData Quality:")
        print(f"  Total trading days: {summary['total_days']}")
        print(f"  News events: {summary['news_events']}")
        print(f"  News days tagged: {summary['news_days_tagged']} ({summary['event_coverage_pct']:.1f}% coverage)")
        print(f"  Valid betas: {summary['valid_betas_pct']:.1f}%")
        print(f"  Average R²: {summary['avg_r_squared']:.3f}")

        print(f"\nAbnormal Returns:")
        print(f"  Mean AR (News): {summary['mean_ar_news']:.6f} ({summary['mean_ar_news']*100:.3f}%)")
        print(f"  Mean AR (Non-News): {summary['mean_ar_non_news']:.6f} ({summary['mean_ar_non_news']*100:.3f}%)")
        print(f"  Difference: {summary['ar_difference']:.6f} ({summary['ar_difference']*100:.3f}%)")

        print(f"\nSentiment Analysis:")
        print(f"  Correlation (ρ): {summary['sentiment_correlation']:.3f}")
        print(f"  P-value: {summary['sentiment_correlation_p']:.4f}")
        print(f"  Significant: {'YES ✓' if summary['sentiment_correlation_p'] < 0.05 else 'NO'}")

        print(f"\nStatistical Tests:")
        print(f"  Significant: {summary['significant_tests']}/{summary['total_tests']}")

        # Save summary
        pd.DataFrame([summary]).to_csv(self.output_dir / "experimental_summary.csv", index=False)

        return summary


def main():
    """Run optimized experimental analysis"""

    # Configuration: test multiple parameter combinations
    experiments = [
        # Balanced approach (recommended)
        {'event_window': (-1, 1), 'polarity_threshold': 0.6, 'name': 'balanced'},
        # Conservative (higher threshold)
        {'event_window': (-1, 1), 'polarity_threshold': 0.7, 'name': 'conservative'},
    ]

    tickers = ['AAPL', 'TSLA']

    all_results = []

    for exp_config in experiments:
        for ticker in tickers:
            print(f"\n{'=' * 80}")
            print(f"RUNNING: {ticker} - {exp_config['name'].upper()} CONFIGURATION")
            print(f"{'=' * 80}\n")

            analysis = ExperimentalAnalysisV3(
                stock_file=f"{ticker}_stock_data.csv",
                news_file=f"{ticker}_eodhd_news.csv",
                ff_file="fama_french_factors.csv",
                ticker=f"{ticker}_{exp_config['name']}",
                data_dir="../01-data",
                output_dir="../03-output",
                event_window=exp_config['event_window'],
                polarity_threshold=exp_config['polarity_threshold']
            )

            try:
                summary = analysis.run_complete_analysis()
                summary['experiment'] = exp_config['name']
                summary['base_ticker'] = ticker
                all_results.append(summary)
                print(f"\n✓ {ticker} - {exp_config['name']} completed successfully!")
            except Exception as e:
                print(f"\n✗ {ticker} - {exp_config['name']} failed: {e}")
                import traceback
                traceback.print_exc()

    # Save comparison of all experiments
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        output_path = Path("../03-output/results/experiment_comparison.csv")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved experiment comparison to {output_path}")


if __name__ == "__main__":
    main()
