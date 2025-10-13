"""
COMPREHENSIVE EVENT STUDY - ALL 17 STOCKS
==========================================

Complete event study analysis for all stocks across all sectors.
Creates individual visualizations and reports like AAPL and TSLA.

Stocks analyzed (17):
- Technology (3): AAPL, MSFT, NVDA
- Finance (2): JPM, GS
- Healthcare (2): JNJ, PFE
- Consumer Discretionary (2): AMZN, TSLA
- Consumer Staples (2): PG, WMT
- Communication Services (2): GOOGL, META
- Energy (1): XOM
- Industrials (1): BA
- Utilities (1): NEE
- Real Estate (1): AMT
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from scipy.stats import ttest_ind, spearmanr
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
statistical_tests_module = importlib.import_module('04_statistical_tests')

DataLoader = data_loader_module.DataLoader
BetaEstimator = beta_estimation_module.BetaEstimator
AbnormalReturnsCalculator = abnormal_returns_module.AbnormalReturnsCalculator
StatisticalTester = statistical_tests_module.StatisticalTester

# Stock configuration
STOCKS = {
    'AAPL': {'sector': 'Technology', 'etf': 'XLK'},
    'MSFT': {'sector': 'Technology', 'etf': 'XLK'},
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'GS': {'sector': 'Finance', 'etf': 'XLF'},
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'TSLA': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC'},
    'META': {'sector': 'Communication Services', 'etf': 'XLC'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'},
    'NEE': {'sector': 'Utilities', 'etf': 'XLU'},
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE'}
}

# Parameters
POLARITY_THRESHOLD = 0.95
EVENT_WINDOW = (0, 0)
DATA_DIR = "../01-data"
OUTPUT_BASE = "../03-output/results"

class ComprehensiveEventStudy:
    """Event study analysis for a single stock"""

    def __init__(self, ticker: str, sector: str, data_dir: str, output_dir: str):
        self.ticker = ticker
        self.sector = sector
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / ticker / "main_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stock_file = f"{ticker}_stock_data.csv"
        self.news_file = f"{ticker}_eodhd_news.csv"
        self.ff_file = "F-F_Research_Data_5_Factors_2x3_daily.csv"

        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # Results
        self.data = None
        self.news_df = None
        self.ar_df = None
        self.beta_df = None
        self.summary = {}

    def run_analysis(self) -> Dict:
        """Run complete event study analysis"""
        print(f"\n{'='*80}")
        print(f"EVENT STUDY: {self.ticker} ({self.sector})")
        print(f"{'='*80}")

        try:
            # Step 1: Load data
            print("\n[1/6] Loading data...")
            self._load_data()

            # Step 2: Estimate betas
            print("[2/6] Estimating factor betas...")
            self._estimate_betas()

            # Step 3: Calculate abnormal returns
            print("[3/6] Calculating abnormal returns...")
            self._calculate_abnormal_returns()

            # Step 4: Tag news days
            print("[4/6] Identifying news days...")
            self._tag_news_days()

            # Step 5: Statistical tests
            print("[5/6] Running statistical tests...")
            self._run_tests()

            # Step 6: Create visualizations
            print("[6/6] Creating visualizations...")
            self._create_visualizations()

            # Generate summary
            self._generate_summary()

            print(f"✅ {self.ticker} analysis complete!")
            return self.summary

        except Exception as e:
            print(f"❌ Error analyzing {self.ticker}: {str(e)}")
            return {'ticker': self.ticker, 'status': 'failed', 'error': str(e)}

    def _load_data(self):
        """Load stock, news, and factor data"""
        # Load stock data directly
        stock_path = self.data_dir / self.stock_file
        stock_df = pd.read_csv(stock_path)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        stock_df.set_index('Date', inplace=True)
        stock_df.sort_index(inplace=True)

        # Calculate returns if not present
        if 'Return' not in stock_df.columns:
            stock_df['Return'] = stock_df['Close'].pct_change()

        # Load Fama-French factors
        ff_path = self.data_dir / self.ff_file
        # Skip header rows and read data
        ff_df = pd.read_csv(ff_path, skiprows=4)  # Skip the description header

        # First column is the date (may be unnamed)
        date_col = ff_df.columns[0]
        ff_df.rename(columns={date_col: 'Date'}, inplace=True)

        # Filter out non-data rows (copyright, etc.) - keep only rows where first column is 8 digits
        ff_df = ff_df[ff_df['Date'].astype(str).str.match(r'^\d{8}$', na=False)].copy()

        # Parse date in YYYYMMDD format
        ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m%d')
        ff_df.set_index('Date', inplace=True)
        ff_df.sort_index(inplace=True)

        # Convert percentages to decimals if needed
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        for col in factor_cols:
            # Remove spaces and convert
            ff_df[col] = pd.to_numeric(ff_df[col], errors='coerce')

        ff_df = ff_df.dropna()

        # Convert to decimals if values are too large (in percentage form)
        for col in factor_cols:
            if ff_df[col].abs().mean() > 1:
                ff_df[col] = ff_df[col] / 100

        # Merge
        self.data = stock_df.join(ff_df, how='inner').dropna()
        self.data['Excess_Return'] = self.data['Return'] - self.data['RF']

        # Load news
        news_path = self.data_dir / self.news_file
        if news_path.exists():
            self.news_df = pd.read_csv(news_path)
            self.news_df['date'] = pd.to_datetime(self.news_df['date'])

            # Filter extreme sentiment
            original_count = len(self.news_df)
            self.news_df = self.news_df[
                abs(self.news_df['sentiment_polarity']) > POLARITY_THRESHOLD
            ].copy()

            # One event per day (strongest sentiment)
            self.news_df['abs_polarity'] = abs(self.news_df['sentiment_polarity'])
            self.news_df = self.news_df.sort_values('abs_polarity', ascending=False)
            self.news_df = self.news_df.groupby(self.news_df['date'].dt.date).first().reset_index(drop=True)

            print(f"  News articles: {original_count:,} → {len(self.news_df)} extreme events")
        else:
            print(f"  ⚠️ No news file found")
            self.news_df = pd.DataFrame()

        print(f"  Stock data: {len(self.data)} days")
        print(f"  Date range: {self.data.index.min().date()} to {self.data.index.max().date()}")

    def _estimate_betas(self):
        """Estimate Fama-French factor betas"""
        estimator = BetaEstimator()
        self.beta_df = estimator.rolling_beta_estimation(
            data=self.data,
            factor_cols=self.factor_cols
        )

        avg_r2 = self.beta_df['R_squared'].mean()
        print(f"  Average R²: {avg_r2:.3f}")

    def _calculate_abnormal_returns(self):
        """Calculate abnormal returns"""
        calculator = AbnormalReturnsCalculator()
        self.ar_df = calculator.calculate_abnormal_returns(
            data=self.data,
            beta_df=self.beta_df,
            factor_cols=self.factor_cols
        )

        print(f"  Mean AR: {self.ar_df['Abnormal_Return'].mean():.4f}")
        print(f"  Std AR: {self.ar_df['Abnormal_Return'].std():.4f}")

    def _tag_news_days(self):
        """Tag trading days with extreme news"""
        self.ar_df['News_Day'] = False
        self.ar_df['Sentiment'] = np.nan

        if len(self.news_df) > 0:
            news_dates = pd.to_datetime(self.news_df['date'].dt.date)

            for date, row in self.news_df.iterrows():
                news_date = pd.to_datetime(row['date'].date())
                if news_date in self.ar_df.index:
                    self.ar_df.loc[news_date, 'News_Day'] = True
                    self.ar_df.loc[news_date, 'Sentiment'] = row['sentiment_polarity']

        news_count = self.ar_df['News_Day'].sum()
        print(f"  News days: {news_count} / {len(self.ar_df)} ({news_count/len(self.ar_df)*100:.1f}%)")

    def _run_tests(self):
        """Run statistical tests"""
        ar_news = self.ar_df[self.ar_df['News_Day']]['Abnormal_Return']
        ar_non_news = self.ar_df[~self.ar_df['News_Day']]['Abnormal_Return']

        # T-test
        t_stat, p_val = ttest_ind(ar_news, ar_non_news, equal_var=False)

        # Correlation (if we have sentiment)
        if self.ar_df['Sentiment'].notna().sum() > 10:
            sentiment_data = self.ar_df[self.ar_df['News_Day']].copy()
            rho, rho_p = spearmanr(sentiment_data['Sentiment'], sentiment_data['Abnormal_Return'])
        else:
            rho, rho_p = np.nan, np.nan

        self.test_results = {
            'mean_ar_news': ar_news.mean(),
            'mean_ar_non_news': ar_non_news.mean(),
            'std_ar_news': ar_news.std(),
            'std_ar_non_news': ar_non_news.std(),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'correlation_rho': rho,
            'correlation_p': rho_p
        }

        print(f"  AR (news): {ar_news.mean():.4f} ± {ar_news.std():.4f}")
        print(f"  AR (non-news): {ar_non_news.mean():.4f} ± {ar_non_news.std():.4f}")
        print(f"  T-test: t={t_stat:.2f}, p={p_val:.4f} {'✓ Significant' if p_val < 0.05 else ''}")
        if not np.isnan(rho):
            print(f"  Correlation: ρ={rho:.3f}, p={rho_p:.4f}")

    def _create_visualizations(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Distribution of ARs
        ax1 = fig.add_subplot(gs[0, 0])
        ar_news = self.ar_df[self.ar_df['News_Day']]['Abnormal_Return']
        ar_non_news = self.ar_df[~self.ar_df['News_Day']]['Abnormal_Return']

        ax1.hist(ar_non_news, bins=50, alpha=0.7, label='Non-News Days', color='blue')
        ax1.hist(ar_news, bins=30, alpha=0.7, label='News Days', color='red')
        ax1.axvline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_xlabel('Abnormal Return')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Abnormal Returns')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Boxplot comparison
        ax2 = fig.add_subplot(gs[0, 1])
        box_data = [
            self.ar_df[~self.ar_df['News_Day']]['Abnormal_Return'],
            self.ar_df[self.ar_df['News_Day']]['Abnormal_Return']
        ]
        bp = ax2.boxplot(box_data, labels=['False', 'True'], patch_artist=True)
        bp['boxes'][0].set_facecolor('blue')
        bp['boxes'][1].set_facecolor('red')
        ax2.set_xlabel('News Day')
        ax2.set_ylabel('Abnormal Return')
        ax2.set_title('AR by News Day')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.grid(alpha=0.3)

        # 3. Time series
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.ar_df.index, self.ar_df['Abnormal_Return'],
                 color='lightblue', alpha=0.6, linewidth=0.5)

        news_days = self.ar_df[self.ar_df['News_Day']]
        ax3.scatter(news_days.index, news_days['Abnormal_Return'],
                   color='red', s=20, alpha=0.6, label='News Days')
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Abnormal Return')
        ax3.set_title('Abnormal Returns Over Time')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Model fit over time
        ax4 = fig.add_subplot(gs[1, 1])
        rolling_r2 = self.beta_df['R_squared'].rolling(window=60, min_periods=20).mean()
        ax4.plot(rolling_r2.index, rolling_r2, color='blue', linewidth=2)
        ax4.axhline(rolling_r2.mean(), color='red', linestyle='--',
                   label=f'Mean: {rolling_r2.mean():.3f}')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('R²')
        ax4.set_title('Model Fit (R²) Over Time')
        ax4.legend()
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 1])

        plt.suptitle(f'{self.ticker} - Event Study Analysis\n{self.sector}',
                    fontsize=16, fontweight='bold', y=0.98)

        # Save
        plt.savefig(self.output_dir / 'analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved: {self.output_dir / 'analysis_summary.png'}")

    def _generate_summary(self):
        """Generate summary statistics"""
        self.summary = {
            'ticker': self.ticker,
            'sector': self.sector,
            'total_days': len(self.ar_df),
            'news_days': self.ar_df['News_Day'].sum(),
            'non_news_days': (~self.ar_df['News_Day']).sum(),
            'mean_ar_news': self.test_results['mean_ar_news'],
            'mean_ar_non_news': self.test_results['mean_ar_non_news'],
            'std_ar_news': self.test_results['std_ar_news'],
            'std_ar_non_news': self.test_results['std_ar_non_news'],
            'avg_r_squared': self.beta_df['R_squared'].mean(),
            'significant_tests': 1 if self.test_results['significant'] else 0,
            'total_tests': 1,
            't_statistic': self.test_results['t_statistic'],
            'p_value': self.test_results['p_value'],
            'correlation_rho': self.test_results['correlation_rho'],
            'correlation_p': self.test_results['correlation_p']
        }

        # Save summary CSV
        summary_df = pd.DataFrame([self.summary])
        summary_df.to_csv(self.output_dir / 'analysis_summary.csv', index=False)

        # Save detailed results
        self.ar_df.to_csv(self.output_dir / 'abnormal_returns.csv')
        self.beta_df.to_csv(self.output_dir / 'beta_estimates.csv')

        print(f"  Saved: {self.output_dir / 'analysis_summary.csv'}")


def main():
    """Run comprehensive event study for all stocks"""
    print("="*100)
    print("COMPREHENSIVE EVENT STUDY - ALL 17 STOCKS")
    print("="*100)
    print(f"\nStocks: {len(STOCKS)}")
    print(f"Sectors: {len(set(s['sector'] for s in STOCKS.values()))}")
    print(f"Polarity threshold: |pol| > {POLARITY_THRESHOLD}")
    print(f"Event window: {EVENT_WINDOW}")

    results = []
    failed = []

    for ticker, info in STOCKS.items():
        study = ComprehensiveEventStudy(
            ticker=ticker,
            sector=info['sector'],
            data_dir=DATA_DIR,
            output_dir=OUTPUT_BASE
        )

        result = study.run_analysis()

        if result.get('status') == 'failed':
            failed.append(ticker)
        else:
            results.append(result)

    # Save aggregate results
    print(f"\n{'='*100}")
    print("AGGREGATE RESULTS")
    print(f"{'='*100}")

    if results:
        results_df = pd.DataFrame(results)
        output_path = Path(OUTPUT_BASE) / "all_stocks_event_study_summary.csv"
        results_df.to_csv(output_path, index=False)

        print(f"\n✅ Successfully analyzed: {len(results)} stocks")
        if failed:
            print(f"❌ Failed: {len(failed)} stocks - {', '.join(failed)}")

        print(f"\nSummary statistics:")
        print(f"  Average news days: {results_df['news_days'].mean():.0f}")
        print(f"  Significant results: {results_df['significant_tests'].sum()} / {len(results)}")
        print(f"  Average R²: {results_df['avg_r_squared'].mean():.3f}")
    else:
        print(f"\n❌ All stocks failed analysis")
        if failed:
            print(f"Failed stocks: {', '.join(failed)}")
        results_df = pd.DataFrame()
        output_path = None

    if output_path:
        print(f"\n📊 Results saved to: {output_path}")
        print(f"📁 Individual reports in: {OUTPUT_BASE}/[TICKER]/main_analysis/")

    return results_df


if __name__ == "__main__":
    results = main()
