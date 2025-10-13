"""
MULTI-SECTOR NEWS DEVIATION ANALYSIS
=====================================

Analyzes news-driven deviations across multiple stocks from different sectors.

KEY CONCEPT: DEVIATION ANALYSIS
Instead of just measuring abnormal returns, we measure how much a stock deviates
from its sector performance on news days. This isolates stock-specific news impact
from sector-wide movements.

Methodology:
1. Calculate stock abnormal return (AR) using Fama-French 5-factor model
2. Calculate sector ETF abnormal return using same model
3. Deviation = Stock AR - Sector AR
4. Compare |deviation| on extreme news days (|pol| > 0.95) vs non-news days

Stocks & Sectors:
- NVDA (Tech) → XLK
- JPM (Finance) → XLF
- PFE (Healthcare) → XLV
- XOM (Energy) → XLE
- AMZN (Consumer Discretionary) → XLY
- BA (Industrials) → XLI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind, spearmanr
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
import importlib
data_loader_module = importlib.import_module('01_data_loader')
beta_estimation_module = importlib.import_module('02_beta_estimation')
abnormal_returns_module = importlib.import_module('03_abnormal_returns')

DataLoader = data_loader_module.DataLoader
BetaEstimator = beta_estimation_module.BetaEstimator
AbnormalReturnsCalculator = abnormal_returns_module.AbnormalReturnsCalculator


# Configuration
STOCK_SECTOR_MAP = {
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'}
}

POLARITY_THRESHOLD = 0.95
DATA_DIR = Path('../01-data')
OUTPUT_DIR = Path('../03-output/results/multi_sector_deviation')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class MultiSectorDeviationAnalysis:
    """Analyzes news-driven deviations across multiple sectors"""

    def __init__(self, stock_sector_map: Dict, polarity_threshold: float = 0.95):
        self.stock_sector_map = stock_sector_map
        self.polarity_threshold = polarity_threshold
        self.factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

        # Results storage
        self.stock_results = {}
        self.comparison_df = None

    def analyze_single_stock(self, ticker: str, sector_etf: str, sector_name: str) -> Dict:
        """Analyze single stock with sector deviation"""
        print(f"\n{'='*80}")
        print(f"ANALYZING: {ticker} ({sector_name}) vs {sector_etf}")
        print(f"{'='*80}")

        # Load data
        print("\n[1/7] Loading data...")
        stock_data, sector_data, ff_data, news_df = self._load_data(ticker, sector_etf)

        if stock_data is None or sector_data is None or news_df is None:
            print(f"  ❌ Insufficient data for {ticker}")
            return None

        # Filter news
        print("\n[2/7] Filtering news...")
        news_filtered = self._filter_news(news_df, stock_data)

        if len(news_filtered) < 10:
            print(f"  ⚠ Only {len(news_filtered)} extreme news events - insufficient for analysis")
            return None

        # Merge data
        print("\n[3/7] Merging stock, sector, and factor data...")
        combined_data = self._merge_data(stock_data, sector_data, ff_data)

        # Calculate abnormal returns
        print("\n[4/7] Calculating stock abnormal returns...")
        stock_ar_df = self._calculate_abnormal_returns(combined_data, ticker_prefix='stock')

        print("\n[5/7] Calculating sector abnormal returns...")
        sector_ar_df = self._calculate_abnormal_returns(combined_data, ticker_prefix='sector')

        # Calculate deviations
        print("\n[6/7] Calculating deviations...")
        deviation_df = self._calculate_deviations(stock_ar_df, sector_ar_df, news_filtered)

        # Analyze deviations
        print("\n[7/7] Running statistical analysis...")
        results = self._analyze_deviations(deviation_df, ticker, sector_name)

        # Save results
        self._save_stock_results(deviation_df, results, ticker)

        return results

    def _load_data(self, ticker: str, sector_etf: str):
        """Load stock, sector, factor, and news data"""
        loader = DataLoader(data_dir=str(DATA_DIR))

        try:
            # Load stock data
            stock_df = loader.load_stock_data(f"{ticker}_stock_data.csv")
            if stock_df is None or len(stock_df) < 100:
                print(f"  ❌ Insufficient stock data for {ticker}")
                return None, None, None, None

            # Load sector ETF data
            sector_df = loader.load_stock_data(f"{sector_etf}_stock_data.csv")
            if sector_df is None or len(sector_df) < 100:
                print(f"  ❌ Insufficient sector data for {sector_etf}")
                return None, None, None, None

            # Load Fama-French factors
            ff_df = loader.load_fama_french_factors('fama_french_factors.csv')
            if ff_df is None or len(ff_df) < 100:
                print(f"  ❌ Insufficient factor data")
                return None, None, None, None

            # Load news
            news_path = DATA_DIR / f"{ticker}_eodhd_news.csv"
            if not news_path.exists():
                print(f"  ❌ News file not found: {news_path}")
                return None, None, None, None

            news_df = pd.read_csv(news_path)
            news_df['date'] = pd.to_datetime(news_df['date'])

            print(f"  ✓ Stock: {len(stock_df)} days")
            print(f"  ✓ Sector: {len(sector_df)} days")
            print(f"  ✓ Factors: {len(ff_df)} days")
            print(f"  ✓ News: {len(news_df)} articles")

            return stock_df, sector_df, ff_df, news_df

        except Exception as e:
            print(f"  ❌ Error loading data: {e}")
            return None, None, None, None

    def _filter_news(self, news_df: pd.DataFrame, stock_data: pd.DataFrame) -> pd.DataFrame:
        """Filter news by polarity threshold and trading days"""
        # Filter by polarity
        news_filtered = news_df[
            abs(news_df['sentiment_polarity']) > self.polarity_threshold
        ].copy()

        # Take strongest sentiment per day
        news_filtered['abs_polarity'] = abs(news_filtered['sentiment_polarity'])
        news_filtered = news_filtered.sort_values('abs_polarity', ascending=False)
        news_filtered = news_filtered.groupby(news_filtered['date'].dt.date).first().reset_index(drop=True)

        # Filter to trading days
        trading_days = pd.DatetimeIndex(stock_data.index).normalize()
        news_dates = pd.to_datetime(news_filtered['date']).dt.tz_localize(None).dt.normalize()
        mask = news_dates.isin(trading_days)
        news_filtered = news_filtered[mask].reset_index(drop=True)

        print(f"  ✓ Filtered to {len(news_filtered)} extreme news events on trading days")
        print(f"  ✓ Polarity range: [{news_filtered['sentiment_polarity'].min():.3f}, {news_filtered['sentiment_polarity'].max():.3f}]")

        return news_filtered

    def _merge_data(self, stock_df: pd.DataFrame, sector_df: pd.DataFrame, ff_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock, sector, and factor data"""
        # Rename columns to avoid conflicts
        stock_df = stock_df.rename(columns={'Return': 'stock_return'})
        sector_df = sector_df.rename(columns={'Return': 'sector_return'})

        # Merge
        combined = stock_df[['stock_return']].join(sector_df[['sector_return']], how='inner')
        combined = combined.join(ff_df, how='inner')

        # Calculate excess returns
        combined['stock_excess_return'] = combined['stock_return'] - combined['RF']
        combined['sector_excess_return'] = combined['sector_return'] - combined['RF']

        print(f"  ✓ Merged data: {len(combined)} days")

        return combined.dropna()

    def _calculate_abnormal_returns(self, data: pd.DataFrame, ticker_prefix: str) -> pd.DataFrame:
        """Calculate abnormal returns using Fama-French 5-factor model"""
        # Prepare data for beta estimation
        temp_data = data.copy()
        temp_data['Excess_Return'] = temp_data[f'{ticker_prefix}_excess_return']
        temp_data['Return'] = temp_data[f'{ticker_prefix}_return']  # Add Return column for calculator

        # Estimate betas
        estimator = BetaEstimator(window_size=126, min_periods=100)
        beta_df = estimator.rolling_beta_estimation(
            data=temp_data,
            factor_cols=self.factor_cols,
            exclude_dates=None,
            event_window=(0, 0)
        )

        # Calculate abnormal returns
        calculator = AbnormalReturnsCalculator()
        ar_df = calculator.calculate_abnormal_returns(
            data=temp_data,
            beta_df=beta_df,
            factor_cols=self.factor_cols
        )

        valid_ar = ar_df['Abnormal_Return'].notna().sum()
        avg_r2 = beta_df['R_squared'].mean()

        print(f"  ✓ Calculated AR for {len(ar_df)} days")
        print(f"  ✓ Valid AR: {valid_ar} ({valid_ar/len(ar_df)*100:.1f}%)")
        print(f"  ✓ Average R²: {avg_r2:.3f}")

        return ar_df

    def _calculate_deviations(self, stock_ar_df: pd.DataFrame,
                             sector_ar_df: pd.DataFrame,
                             news_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate deviations from sector"""
        # Combine stock and sector ARs
        deviation_df = pd.DataFrame({
            'stock_ar': stock_ar_df['Abnormal_Return'],
            'sector_ar': sector_ar_df['Abnormal_Return']
        })

        # Calculate deviation
        deviation_df['deviation'] = deviation_df['stock_ar'] - deviation_df['sector_ar']
        deviation_df['abs_deviation'] = abs(deviation_df['deviation'])

        # Tag news days
        deviation_df['news_day'] = False
        deviation_df['news_sentiment'] = np.nan

        news_df['date_normalized'] = pd.to_datetime(news_df['date']).dt.tz_localize(None).dt.normalize()

        for _, news_row in news_df.iterrows():
            news_date = news_row['date_normalized']
            if news_date in deviation_df.index:
                deviation_df.loc[news_date, 'news_day'] = True
                deviation_df.loc[news_date, 'news_sentiment'] = news_row['sentiment_polarity']

        news_count = deviation_df['news_day'].sum()
        print(f"  ✓ Deviation calculated for {len(deviation_df)} days")
        print(f"  ✓ News days: {news_count} ({news_count/len(deviation_df)*100:.1f}%)")

        return deviation_df.dropna(subset=['deviation'])

    def _analyze_deviations(self, deviation_df: pd.DataFrame, ticker: str, sector: str) -> Dict:
        """Analyze deviation statistics"""
        news_dev = deviation_df[deviation_df['news_day'] == True]['abs_deviation'].dropna()
        non_news_dev = deviation_df[deviation_df['news_day'] == False]['abs_deviation'].dropna()

        if len(news_dev) < 5 or len(non_news_dev) < 10:
            print(f"  ⚠ Insufficient data for statistical tests")
            return None

        # T-test
        t_stat, p_val = ttest_ind(news_dev, non_news_dev)

        # Correlation
        news_data = deviation_df[deviation_df['news_day'] == True]
        if len(news_data) > 2:
            rho, rho_p = spearmanr(news_data['news_sentiment'], news_data['deviation'])
        else:
            rho, rho_p = np.nan, np.nan

        results = {
            'ticker': ticker,
            'sector': sector,
            'total_days': len(deviation_df),
            'news_days': len(news_dev),
            'mean_dev_news': news_dev.mean(),
            'std_dev_news': news_dev.std(),
            'mean_dev_non_news': non_news_dev.mean(),
            'std_dev_non_news': non_news_dev.std(),
            'deviation_increase_pct': ((news_dev.mean() - non_news_dev.mean()) / non_news_dev.mean() * 100),
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'correlation_rho': rho,
            'correlation_p': rho_p
        }

        print(f"\n  📊 DEVIATION ANALYSIS:")
        print(f"    News days: |deviation| = {results['mean_dev_news']:.4f} ({results['mean_dev_news']*100:.2f}%)")
        print(f"    Non-news:  |deviation| = {results['mean_dev_non_news']:.4f} ({results['mean_dev_non_news']*100:.2f}%)")
        print(f"    Increase: {results['deviation_increase_pct']:.1f}%")
        print(f"    T-test: t={t_stat:.3f}, p={p_val:.4f} {'✓ SIGNIFICANT' if p_val < 0.05 else '✗ Not significant'}")
        print(f"    Correlation: ρ={rho:.3f}, p={rho_p:.4f}")

        return results

    def _save_stock_results(self, deviation_df: pd.DataFrame, results: Dict, ticker: str):
        """Save individual stock results"""
        stock_dir = OUTPUT_DIR / ticker
        stock_dir.mkdir(exist_ok=True, parents=True)

        # Save deviation data
        deviation_df.to_csv(stock_dir / 'deviations.csv')

        # Save summary
        if results:
            pd.DataFrame([results]).to_csv(stock_dir / 'summary.csv', index=False)

        print(f"  ✓ Saved results to {stock_dir}")

    def run_multi_sector_analysis(self) -> pd.DataFrame:
        """Run analysis across all sectors"""
        print("=" * 80)
        print("MULTI-SECTOR DEVIATION ANALYSIS")
        print("=" * 80)
        print(f"\nPolarity Threshold: |polarity| > {self.polarity_threshold}")
        print(f"Stocks: {len(self.stock_sector_map)}")
        print()

        all_results = []

        for ticker, info in self.stock_sector_map.items():
            try:
                result = self.analyze_single_stock(ticker, info['etf'], info['sector'])
                if result:
                    all_results.append(result)
                    self.stock_results[ticker] = result
            except Exception as e:
                print(f"\n  ❌ {ticker} failed: {e}")
                import traceback
                traceback.print_exc()

        # Create comparison dataframe
        if all_results:
            self.comparison_df = pd.DataFrame(all_results)
            self.comparison_df = self.comparison_df.sort_values('deviation_increase_pct', ascending=False)

            # Save comparison
            self.comparison_df.to_csv(OUTPUT_DIR / 'sector_comparison.csv', index=False)

            # Generate comparison report
            self._generate_comparison_report()

            # Create visualizations
            self._create_visualizations()

        return self.comparison_df

    def _generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("CROSS-SECTOR COMPARISON")
        print("=" * 80)

        if self.comparison_df is None or len(self.comparison_df) == 0:
            print("No results to compare")
            return

        print(f"\nStocks Analyzed: {len(self.comparison_df)}")
        print(f"Significant Results: {self.comparison_df['significant'].sum()}/{len(self.comparison_df)}")

        print("\n" + "-" * 80)
        print("DEVIATION INCREASE ON NEWS DAYS (Ranked)")
        print("-" * 80)

        for _, row in self.comparison_df.iterrows():
            sig = "✓" if row['significant'] else "✗"
            print(f"{row['ticker']:6} ({row['sector']:23}): {row['deviation_increase_pct']:+6.1f}% {sig}")

        print("\n" + "-" * 80)
        print("DETAILED STATISTICS")
        print("-" * 80)
        print(self.comparison_df[['ticker', 'sector', 'news_days', 'mean_dev_news',
                                  'mean_dev_non_news', 'p_value', 'significant']].to_string(index=False))

        # Summary stats
        print("\n" + "-" * 80)
        print("SUMMARY STATISTICS")
        print("-" * 80)
        print(f"Average deviation increase: {self.comparison_df['deviation_increase_pct'].mean():.1f}%")
        print(f"Median deviation increase: {self.comparison_df['deviation_increase_pct'].median():.1f}%")
        print(f"Max deviation increase: {self.comparison_df['deviation_increase_pct'].max():.1f}% ({self.comparison_df.loc[self.comparison_df['deviation_increase_pct'].idxmax(), 'ticker']})")

    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.comparison_df is None or len(self.comparison_df) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        sns.set_style("whitegrid")

        # 1. Deviation increase by sector
        ax = axes[0, 0]
        colors = ['green' if x else 'red' for x in self.comparison_df['significant']]
        ax.barh(self.comparison_df['ticker'], self.comparison_df['deviation_increase_pct'], color=colors, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Deviation Increase on News Days (%)')
        ax.set_title('News Impact by Stock (✓ = significant)')
        ax.grid(alpha=0.3)

        # 2. Mean deviation comparison
        ax = axes[0, 1]
        x = np.arange(len(self.comparison_df))
        width = 0.35
        ax.bar(x - width/2, self.comparison_df['mean_dev_news'], width, label='News Days', alpha=0.7)
        ax.bar(x + width/2, self.comparison_df['mean_dev_non_news'], width, label='Non-News Days', alpha=0.7)
        ax.set_xlabel('Stock')
        ax.set_ylabel('Mean Absolute Deviation')
        ax.set_title('Deviation Magnitude Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(self.comparison_df['ticker'])
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. P-value distribution
        ax = axes[0, 2]
        ax.scatter(self.comparison_df['ticker'], self.comparison_df['p_value'], s=100, alpha=0.7)
        ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax.set_ylabel('P-value')
        ax.set_title('Statistical Significance')
        ax.set_ylim([0, max(0.1, self.comparison_df['p_value'].max() * 1.1)])
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Sector breakdown
        ax = axes[1, 0]
        sector_counts = self.comparison_df['sector'].value_counts()
        ax.pie(sector_counts, labels=sector_counts.index, autopct='%1.0f%%', startangle=90)
        ax.set_title('Sectors Analyzed')

        # 5. News days distribution
        ax = axes[1, 1]
        ax.bar(self.comparison_df['ticker'], self.comparison_df['news_days'], alpha=0.7)
        ax.set_ylabel('Number of Extreme News Days')
        ax.set_title('Sample Size by Stock')
        ax.grid(alpha=0.3)

        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = f"""
MULTI-SECTOR ANALYSIS SUMMARY
{'='*35}

Stocks Analyzed: {len(self.comparison_df)}
Sectors: {self.comparison_df['sector'].nunique()}

DEVIATION INCREASE:
  Average: {self.comparison_df['deviation_increase_pct'].mean():.1f}%
  Median: {self.comparison_df['deviation_increase_pct'].median():.1f}%
  Range: [{self.comparison_df['deviation_increase_pct'].min():.1f}%,
         {self.comparison_df['deviation_increase_pct'].max():.1f}%]

STATISTICAL SIGNIFICANCE:
  Significant: {self.comparison_df['significant'].sum()}/{len(self.comparison_df)}
  ({self.comparison_df['significant'].sum()/len(self.comparison_df)*100:.0f}%)

TOP STOCK:
  {self.comparison_df.iloc[0]['ticker']} ({self.comparison_df.iloc[0]['sector']})
  +{self.comparison_df.iloc[0]['deviation_increase_pct']:.1f}%
  p = {self.comparison_df.iloc[0]['p_value']:.4f}
"""
        ax.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
               verticalalignment='center')

        plt.suptitle('Multi-Sector News Deviation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'multi_sector_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n  ✓ Saved visualization to {OUTPUT_DIR / 'multi_sector_comparison.png'}")


def main():
    """Run multi-sector analysis"""
    analyzer = MultiSectorDeviationAnalysis(
        stock_sector_map=STOCK_SECTOR_MAP,
        polarity_threshold=POLARITY_THRESHOLD
    )

    results_df = analyzer.run_multi_sector_analysis()

    print("\n" + "=" * 80)
    print("✅ MULTI-SECTOR ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}")

    return results_df


if __name__ == "__main__":
    main()
