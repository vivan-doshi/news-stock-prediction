"""
CATEGORY-SPECIFIC EVENT STUDY ANALYSIS
=======================================

Performs event study analysis for each news category separately.
Tests which types of news have significant impact on stock returns.

Structure:
- For each stock: run event study for each of 8 categories
- For each sector: aggregate results across stocks within sector
- Overall: compare category effectiveness across all stocks/sectors

Methodology:
- Fama-French 5-Factor model for abnormal returns
- Winsorization for outlier control
- Bootstrap confidence intervals
- Multiple statistical tests (t-test, Mann-Whitney U)
- Cohen's d for effect size

Output:
- Individual: [TICKER]/[CATEGORY]/event_study_results.csv
- Sector: by_sector/[SECTOR]/category_comparison.png
- Overall: comprehensive_category_results.csv

Author: Category Event Study System
Date: 2025-10-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats.mstats import winsorize
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import custom modules
import importlib
data_loader_module = importlib.import_module('01_data_loader')
beta_estimation_module = importlib.import_module('02_beta_estimation')
abnormal_returns_module = importlib.import_module('03_abnormal_returns')

DataLoader = data_loader_module.DataLoader
BetaEstimator = beta_estimation_module.BetaEstimator
AbnormalReturnsCalculator = abnormal_returns_module.AbnormalReturnsCalculator

# Import stock configuration
config_module = importlib.import_module('21_expanded_50_stock_config')
STOCKS = config_module.EXPANDED_STOCKS

# Parameters
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = str(SCRIPT_DIR.parent / "01-data")
EVENT_DATES_DIR = SCRIPT_DIR.parent / "03-output" / "category_event_study" / "event_dates"
OUTPUT_DIR = SCRIPT_DIR.parent / "03-output" / "category_event_study" / "results"

# Statistical parameters
WINSORIZE_LIMIT = 0.01
N_BOOTSTRAP = 1000
RANDOM_SEED = 42
MIN_EVENTS = 5  # Minimum events required to run analysis for a category

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

np.random.seed(RANDOM_SEED)

# Categories
CATEGORIES = [
    'Earnings',
    'Product_Launch',
    'Executive_Changes',
    'M&A',
    'Regulatory_Legal',
    'Analyst_Ratings',
    'Dividends',
    'Market_Performance'
]


class CategoryEventStudy:
    """Event study analysis for a specific news category"""

    def __init__(self, ticker: str, category: str, sector: str, data_dir: str, output_dir: Path):
        self.ticker = ticker
        self.category = category
        self.sector = sector
        self.data_dir = Path(data_dir)
        self.output_dir = output_dir / ticker / category
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stock_file = f"{ticker}_stock_data.csv"
        self.ff_file = "F-F_Research_Data_5_Factors_2x3_daily.csv"
        self.event_file = f"{category}_events.csv"

        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # Results
        self.data = None
        self.event_dates = []
        self.ar_df = None
        self.beta_df = None
        self.summary = {}

    def run_analysis(self) -> Dict:
        """Run complete category event study"""
        print(f"\n  Category: {self.category}")

        try:
            # Load data
            self._load_data()

            if self.data is None or len(self.data) < 100:
                return self._create_insufficient_data_result()

            if len(self.event_dates) < MIN_EVENTS:
                print(f"    ⚠️  Only {len(self.event_dates)} events (min {MIN_EVENTS} required)")
                return self._create_insufficient_events_result()

            # Estimate betas
            self._estimate_betas()

            # Calculate abnormal returns
            self._calculate_abnormal_returns()

            # Tag category event days
            self._tag_event_days()

            # Run statistical tests
            self._run_statistical_tests()

            # Bootstrap CI
            self._compute_bootstrap_ci()

            # Create visualization
            self._create_visualization()

            # Generate summary
            self._generate_summary()

            print(f"    ✅ Complete: {len(self.event_dates)} events analyzed")
            return self.summary

        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
            return {
                'ticker': self.ticker,
                'category': self.category,
                'sector': self.sector,
                'status': 'failed',
                'error': str(e)
            }

    def _load_data(self):
        """Load stock data, Fama-French factors, and category event dates"""
        # Load stock data
        stock_path = self.data_dir / self.stock_file
        if not stock_path.exists():
            raise FileNotFoundError(f"Stock file not found: {stock_path}")

        stock_df = pd.read_csv(stock_path)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df.set_index('Date', inplace=True)
        if hasattr(stock_df.index, 'tz') and stock_df.index.tz is not None:
            stock_df.index = stock_df.index.tz_localize(None)
        stock_df.sort_index(inplace=True)

        if 'Return' not in stock_df.columns:
            stock_df['Return'] = stock_df['Close'].pct_change()

        stock_df = stock_df[np.abs(stock_df['Return']) < 0.5].copy()

        # Load Fama-French factors
        ff_path = self.data_dir / self.ff_file
        ff_df = pd.read_csv(ff_path, skiprows=4)

        date_col = ff_df.columns[0]
        ff_df.rename(columns={date_col: 'Date'}, inplace=True)
        ff_df = ff_df[ff_df['Date'].astype(str).str.match(r'^\d{8}$', na=False)].copy()
        ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
        ff_df.set_index('Date', inplace=True)
        ff_df.sort_index(inplace=True)

        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for col in factor_cols:
            ff_df[col] = pd.to_numeric(ff_df[col], errors='coerce')

        ff_df[factor_cols] = ff_df[factor_cols].fillna(method='ffill').fillna(method='bfill')

        for col in factor_cols:
            if ff_df[col].abs().mean() > 1:
                ff_df[col] = ff_df[col] / 100

        # Merge
        self.data = stock_df.join(ff_df, how='inner')
        self.data = self.data.dropna()
        self.data['Excess_Return'] = self.data['Return'] - self.data['RF']

        # Load category event dates
        event_path = EVENT_DATES_DIR / self.ticker / self.event_file
        if event_path.exists():
            events_df = pd.read_csv(event_path)
            try:
                date_series = pd.to_datetime(events_df['event_date'], errors='coerce')
                date_series = date_series.dropna()
                self.event_dates = date_series.dt.date.unique()
            except Exception as e:
                print(f"      ⚠️ Error parsing dates: {e}")
                self.event_dates = []
        else:
            self.event_dates = []

        print(f"    Data: {len(self.data)} days, Events: {len(self.event_dates)}")

    def _estimate_betas(self):
        """Estimate factor betas"""
        estimator = BetaEstimator()
        self.beta_df = estimator.rolling_beta_estimation(
            data=self.data,
            factor_cols=self.factor_cols
        )

        beta_cols = [col for col in self.beta_df.columns if col.startswith('beta_') or col.startswith('Beta_')]
        self.beta_df[beta_cols] = self.beta_df[beta_cols].fillna(method='ffill').fillna(method='bfill')
        self.beta_df = self.beta_df.fillna(0)

    def _calculate_abnormal_returns(self):
        """Calculate abnormal returns with winsorization"""
        calculator = AbnormalReturnsCalculator()
        self.ar_df = calculator.calculate_abnormal_returns(
            data=self.data,
            beta_df=self.beta_df,
            factor_cols=self.factor_cols
        )

        self.ar_df = self.ar_df.dropna(subset=['Abnormal_Return'])

        ar_values = self.ar_df['Abnormal_Return'].values
        ar_winsorized = winsorize(ar_values, limits=[WINSORIZE_LIMIT, WINSORIZE_LIMIT])
        self.ar_df['AR_Winsorized'] = ar_winsorized

    def _tag_event_days(self):
        """Tag trading days with category events"""
        self.ar_df['Event_Day'] = False

        if len(self.event_dates) > 0:
            for event_date in self.event_dates:
                event_datetime = pd.Timestamp(event_date)
                if event_datetime in self.ar_df.index:
                    self.ar_df.loc[event_datetime, 'Event_Day'] = True

        event_count = self.ar_df['Event_Day'].sum()
        print(f"    Event days matched: {event_count} / {len(self.event_dates)}")

    def _run_statistical_tests(self):
        """Run statistical tests"""
        ar_event = self.ar_df[self.ar_df['Event_Day']]['AR_Winsorized']
        ar_non_event = self.ar_df[~self.ar_df['Event_Day']]['AR_Winsorized']

        if len(ar_event) < 2:
            self.test_results = {
                'mean_ar_event': np.nan,
                'mean_ar_non_event': ar_non_event.mean() if len(ar_non_event) > 0 else np.nan,
                'std_ar_event': np.nan,
                'std_ar_non_event': ar_non_event.std() if len(ar_non_event) > 0 else np.nan,
                't_statistic': np.nan,
                'p_value_ttest': np.nan,
                'p_value_mannwhitney': np.nan,
                'significant': False,
                'cohens_d': np.nan
            }
            return

        # T-test
        t_stat, p_ttest = ttest_ind(ar_event, ar_non_event, equal_var=False)

        # Mann-Whitney U
        try:
            u_stat, p_mannwhitney = mannwhitneyu(ar_event, ar_non_event, alternative='two-sided')
        except:
            p_mannwhitney = np.nan

        # Cohen's d
        pooled_std = np.sqrt((ar_event.std()**2 + ar_non_event.std()**2) / 2)
        cohens_d = (ar_event.mean() - ar_non_event.mean()) / pooled_std if pooled_std > 0 else np.nan

        self.test_results = {
            'mean_ar_event': ar_event.mean(),
            'mean_ar_non_event': ar_non_event.mean(),
            'median_ar_event': ar_event.median(),
            'median_ar_non_event': ar_non_event.median(),
            'std_ar_event': ar_event.std(),
            'std_ar_non_event': ar_non_event.std(),
            't_statistic': t_stat,
            'p_value_ttest': p_ttest,
            'p_value_mannwhitney': p_mannwhitney,
            'significant': p_ttest < 0.05,
            'cohens_d': cohens_d
        }

    def _compute_bootstrap_ci(self):
        """Compute bootstrap confidence intervals"""
        ar_event = self.ar_df[self.ar_df['Event_Day']]['AR_Winsorized']
        ar_non_event = self.ar_df[~self.ar_df['Event_Day']]['AR_Winsorized']

        if len(ar_event) < 2:
            self.bootstrap_ci = {
                'event_ci_lower': np.nan,
                'event_ci_upper': np.nan,
                'non_event_ci_lower': np.nan,
                'non_event_ci_upper': np.nan
            }
            return

        event_means = [np.random.choice(ar_event, size=len(ar_event), replace=True).mean()
                      for _ in range(N_BOOTSTRAP)]
        non_event_means = [np.random.choice(ar_non_event, size=len(ar_non_event), replace=True).mean()
                          for _ in range(N_BOOTSTRAP)]

        self.bootstrap_ci = {
            'event_ci_lower': np.percentile(event_means, 2.5),
            'event_ci_upper': np.percentile(event_means, 97.5),
            'non_event_ci_lower': np.percentile(non_event_means, 2.5),
            'non_event_ci_upper': np.percentile(non_event_means, 97.5)
        }

    def _create_visualization(self):
        """Create compact visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ar_event = self.ar_df[self.ar_df['Event_Day']]['AR_Winsorized']
        ar_non_event = self.ar_df[~self.ar_df['Event_Day']]['AR_Winsorized']

        # Histogram
        ax1 = axes[0]
        ax1.hist(ar_non_event, bins=40, alpha=0.6, label='Non-Event Days', color='steelblue', density=True)
        if len(ar_event) > 0:
            ax1.hist(ar_event, bins=20, alpha=0.7, label='Event Days', color='coral', density=True)
        ax1.axvline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Abnormal Return', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title(f'{self.category} Event Distribution', fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Boxplot
        ax2 = axes[1]
        if len(ar_event) > 0:
            box_data = [ar_non_event, ar_event]
            bp = ax2.boxplot(box_data, labels=['Non-Event', 'Event'], patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('steelblue')
            bp['boxes'][1].set_facecolor('coral')
        else:
            box_data = [ar_non_event]
            bp = ax2.boxplot(box_data, labels=['Non-Event'], patch_artist=True, showmeans=True)
            bp['boxes'][0].set_facecolor('steelblue')

        ax2.set_ylabel('Abnormal Return', fontweight='bold')
        ax2.set_title('AR Comparison', fontweight='bold')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(alpha=0.3, axis='y')

        # Main title
        if hasattr(self, 'test_results') and not np.isnan(self.test_results['p_value_ttest']):
            sig = '***' if self.test_results['p_value_ttest'] < 0.001 else \
                  '**' if self.test_results['p_value_ttest'] < 0.01 else \
                  '*' if self.test_results['p_value_ttest'] < 0.05 else 'ns'
            title = f'{self.ticker} - {self.category} ({self.sector})\n' + \
                    f'Event AR={self.test_results["mean_ar_event"]:.4f}, Non-Event={self.test_results["mean_ar_non_event"]:.4f} ' + \
                    f'| p={self.test_results["p_value_ttest"]:.4f} {sig} | d={self.test_results["cohens_d"]:.3f}'
        else:
            title = f'{self.ticker} - {self.category} ({self.sector})'

        plt.suptitle(title, fontsize=11, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / 'event_study.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_summary(self):
        """Generate summary statistics"""
        self.summary = {
            'ticker': self.ticker,
            'category': self.category,
            'sector': self.sector,
            'status': 'success',
            'num_events': self.ar_df['Event_Day'].sum(),
            'total_days': len(self.ar_df),
            'mean_ar_event': self.test_results.get('mean_ar_event', np.nan),
            'mean_ar_non_event': self.test_results.get('mean_ar_non_event', np.nan),
            'median_ar_event': self.test_results.get('median_ar_event', np.nan),
            'median_ar_non_event': self.test_results.get('median_ar_non_event', np.nan),
            'std_ar_event': self.test_results.get('std_ar_event', np.nan),
            'std_ar_non_event': self.test_results.get('std_ar_non_event', np.nan),
            't_statistic': self.test_results.get('t_statistic', np.nan),
            'p_value_ttest': self.test_results.get('p_value_ttest', np.nan),
            'p_value_mannwhitney': self.test_results.get('p_value_mannwhitney', np.nan),
            'significant': self.test_results.get('significant', False),
            'cohens_d': self.test_results.get('cohens_d', np.nan),
            'avg_r_squared': self.beta_df['R_squared'].mean(),
            'ci_event_lower': self.bootstrap_ci.get('event_ci_lower', np.nan) if hasattr(self, 'bootstrap_ci') else np.nan,
            'ci_event_upper': self.bootstrap_ci.get('event_ci_upper', np.nan) if hasattr(self, 'bootstrap_ci') else np.nan,
        }

        # Save summary
        summary_df = pd.DataFrame([self.summary])
        summary_df.to_csv(self.output_dir / 'summary.csv', index=False)

    def _create_insufficient_data_result(self):
        return {
            'ticker': self.ticker,
            'category': self.category,
            'sector': self.sector,
            'status': 'insufficient_data',
            'num_events': 0,
            'total_days': len(self.data) if self.data is not None else 0
        }

    def _create_insufficient_events_result(self):
        return {
            'ticker': self.ticker,
            'category': self.category,
            'sector': self.sector,
            'status': 'insufficient_events',
            'num_events': len(self.event_dates),
            'total_days': len(self.data) if self.data is not None else 0
        }


def run_stock_analysis(ticker: str, sector: str) -> List[Dict]:
    """Run analysis for all categories for a given stock"""
    print(f"\n{'='*70}")
    print(f"{ticker} ({sector})")
    print(f"{'='*70}")

    results = []

    for category in CATEGORIES:
        study = CategoryEventStudy(
            ticker=ticker,
            category=category,
            sector=sector,
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR
        )

        result = study.run_analysis()
        results.append(result)

    return results


def main():
    """Run category event study for all stocks"""
    print("="*80)
    print("CATEGORY-SPECIFIC EVENT STUDY")
    print("="*80)
    print(f"\nStocks: {len(STOCKS)}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"  {', '.join(CATEGORIES)}")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Process each stock
    for i, (ticker, info) in enumerate(STOCKS.items(), 1):
        print(f"\n[{i}/{len(STOCKS)}] {ticker}...")

        stock_results = run_stock_analysis(ticker, info['sector'])
        all_results.extend(stock_results)

    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR.parent / "comprehensive_category_results.csv"
    results_df.to_csv(results_file, index=False)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    successful = results_df[results_df['status'] == 'success']
    print(f"\n✅ Successfully analyzed: {len(successful)} category-stock combinations")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print(f"📊 Comprehensive results: {results_file}")

    return results_df


if __name__ == "__main__":
    results_df = main()