"""
Comprehensive News Filtering System
Implements multiple filtering strategies with side-by-side comparison
Addresses false positives, multi-ticker articles, and duplicates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import stock configuration
import sys
sys.path.append(str(Path(__file__).parent))
# Use the underscore prefix version
import importlib.util
spec = importlib.util.spec_from_file_location("config", Path(__file__).parent / "21_expanded_50_stock_config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
EXPANDED_STOCKS = config_module.EXPANDED_STOCKS

# Set up paths
DATA_DIR = Path(__file__).parent.parent / "01-data"
OUTPUT_DIR = Path(__file__).parent.parent / "03-output" / "news_filtering_comparison"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class ComprehensiveNewsFilter:
    """Advanced news filtering with multiple strategies and false positive detection"""

    def __init__(self, ticker, sector, company_name):
        self.ticker = ticker
        self.sector = sector
        self.company_name = company_name
        self.df = None
        self.results = {}

        # Event category definitions
        self.event_categories = {
            'Earnings': ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4',
                        'revenue', 'profit', 'eps', 'beats estimates', 'misses estimates'],
            'Product Launch': ['launch', 'release', 'unveil', 'introduce', 'new product', 'new model',
                             'announces', 'unveiled', 'releasing'],
            'Executive Changes': ['ceo', 'executive', 'leadership', 'management', 'board', 'resign',
                                'appoint', 'hire', 'chief', 'president', 'director'],
            'M&A': ['merger', 'acquisition', 'acquire', 'acquiring', 'buyout', 'takeover',
                   'joint venture', 'strategic partnership'],
            'Regulatory/Legal': ['sec', 'lawsuit', 'investigation', 'probe', 'regulator', 'compliance',
                               'legal', 'fine', 'settlement', 'court', 'judge'],
            'Analyst Ratings': ['upgrade', 'downgrade', 'rating', 'target price', 'analyst',
                              'price target', 'outperform', 'underperform', 'buy', 'sell'],
            'Dividends': ['dividend', 'payout', 'distribution', 'yield', 'dividend increase'],
            'Market Performance': ['stock', 'shares', 'market', 'trading', 'price', 'rally',
                                 'drop', 'surge', 'plunge', 'investors', 'wall street']
        }

    def load_data(self):
        """Load news data for the ticker"""
        file_path = DATA_DIR / f"{self.ticker}_eodhd_news.csv"

        if not file_path.exists():
            print(f"  ⚠️  No news data found for {self.ticker}")
            return False

        self.df = pd.read_csv(file_path)
        self.df['date'] = pd.to_datetime(self.df['date'])

        print(f"  ✓ Loaded {len(self.df):,} articles for {self.ticker}")
        return True

    def detect_false_positives(self):
        """Detect various false positive indicators"""
        print(f"  → Analyzing false positive indicators...")

        # 1. Ticker in title (case insensitive)
        self.df['ticker_in_title'] = self.df['title'].str.contains(
            self.ticker, case=False, na=False, regex=False
        )

        # Also check for company name in title
        company_keywords = self.company_name.split()[:2]  # First 2 words
        for keyword in company_keywords:
            if len(keyword) > 3:  # Skip short words
                self.df['ticker_in_title'] |= self.df['title'].str.contains(
                    keyword, case=False, na=False, regex=False
                )

        # 2. Count tickers mentioned (parse symbols field)
        def count_tickers(symbols_str):
            if pd.isna(symbols_str):
                return 1
            # Symbols are comma-separated
            tickers = [s.strip() for s in str(symbols_str).split(',')]
            return len([t for t in tickers if len(t) > 0])

        self.df['ticker_count'] = self.df['symbols'].apply(count_tickers)

        # 3. Content quality
        self.df['content_length'] = self.df['content'].astype(str).str.len()
        self.df['title_length'] = self.df['title'].astype(str).str.len()

        # 4. Duplicate detection (within this ticker's data)
        self.df['is_duplicate'] = self.df.duplicated(subset=['title'], keep='first')

        # 5. False positive score (0 = best, 3 = worst)
        self.df['fp_score'] = 0
        self.df.loc[~self.df['ticker_in_title'], 'fp_score'] += 1
        self.df.loc[self.df['ticker_count'] > 2, 'fp_score'] += 1
        self.df.loc[self.df['content_length'] < 200, 'fp_score'] += 1

        # Print statistics
        print(f"    • Ticker in title: {self.df['ticker_in_title'].sum():,} ({self.df['ticker_in_title'].sum()/len(self.df)*100:.1f}%)")
        print(f"    • Single ticker: {(self.df['ticker_count'] == 1).sum():,} ({(self.df['ticker_count'] == 1).sum()/len(self.df)*100:.1f}%)")
        print(f"    • Content ≥200 chars: {(self.df['content_length'] >= 200).sum():,} ({(self.df['content_length'] >= 200).sum()/len(self.df)*100:.1f}%)")
        print(f"    • Duplicates: {self.df['is_duplicate'].sum():,} ({self.df['is_duplicate'].sum()/len(self.df)*100:.1f}%)")
        print(f"    • FP Score 0 (clean): {(self.df['fp_score'] == 0).sum():,} ({(self.df['fp_score'] == 0).sum()/len(self.df)*100:.1f}%)")

    def categorize_events(self):
        """Categorize articles by event type"""
        print(f"  → Categorizing events...")

        for category, keywords in self.event_categories.items():
            pattern = '|'.join([f'\\b{kw}\\b' for kw in keywords])

            # Check in title (higher weight) and content
            title_match = self.df['title'].str.contains(pattern, case=False, na=False, regex=True)
            content_match = self.df['content'].str.contains(pattern, case=False, na=False, regex=True)

            self.df[f'cat_{category}'] = title_match | content_match

        # Count categories per article
        cat_cols = [f'cat_{cat}' for cat in self.event_categories.keys()]
        self.df['num_categories'] = self.df[cat_cols].sum(axis=1)

        # Assign primary category (first match in priority order)
        priority_order = ['Earnings', 'Product Launch', 'Regulatory/Legal', 'Analyst Ratings',
                         'Executive Changes', 'Dividends', 'M&A', 'Market Performance']

        def get_primary_category(row):
            for cat in priority_order:
                if row[f'cat_{cat}']:
                    return cat
            return 'Uncategorized'

        self.df['primary_category'] = self.df.apply(get_primary_category, axis=1)

        # Print distribution
        cat_dist = self.df['primary_category'].value_counts()
        print(f"    • Categorized: {(self.df['num_categories'] > 0).sum():,} ({(self.df['num_categories'] > 0).sum()/len(self.df)*100:.1f}%)")

    def apply_strategy_precision(self):
        """Strategy A: Precision-Focused (Low False Positives)"""
        mask = (
            (self.df['ticker_in_title']) &
            (self.df['ticker_count'] <= 2) &
            (self.df['content_length'] >= 200) &
            (self.df['primary_category'].isin(['Earnings', 'Product Launch', 'Regulatory/Legal', 'Analyst Ratings'])) &
            (~self.df['is_duplicate'])
        )

        filtered = self.df[mask].copy()

        # One article per day (highest sentiment magnitude)
        if len(filtered) > 0:
            filtered['sentiment_abs'] = filtered['sentiment_polarity'].abs()
            filtered = filtered.sort_values(['date', 'sentiment_abs'], ascending=[True, False])
            filtered = filtered.groupby(filtered['date'].dt.date).first().reset_index(drop=True)

        return filtered

    def apply_strategy_recall(self):
        """Strategy B: Recall-Focused (Comprehensive Coverage)"""
        mask = (
            (
                (self.df['ticker_in_title']) |
                (self.df['sentiment_polarity'].abs() > 0.7)
            ) &
            (self.df['ticker_count'] <= 5) &
            (self.df['content_length'] >= 100) &
            (self.df['num_categories'] > 0)  # Must have at least one category
        )

        return self.df[mask].copy()

    def apply_strategy_balanced(self):
        """Strategy C: Balanced (Recommended)"""
        # Either ticker in title, OR (<=2 tickers AND extreme sentiment)
        condition1 = self.df['ticker_in_title']
        condition2 = (self.df['ticker_count'] <= 2) & (self.df['sentiment_polarity'].abs() > 0.6)

        mask = (
            (condition1 | condition2) &
            (self.df['content_length'] >= 200) &
            (self.df['primary_category'].isin(['Earnings', 'Product Launch', 'Regulatory/Legal',
                                               'Analyst Ratings', 'Executive Changes', 'Dividends']))
        )

        # Deduplicate within stock-date pairs
        filtered = self.df[mask].copy()
        if len(filtered) > 0:
            filtered = filtered.drop_duplicates(subset=['date', 'title'], keep='first')

        return filtered

    def apply_strategy_category_specific(self):
        """Strategy D: Category-Specific (Different rules per event type)"""
        results = []

        # Earnings: ticker in title + keyword match, any sentiment, <=3 tickers
        earnings_mask = (
            (self.df['cat_Earnings']) &
            (self.df['ticker_in_title']) &
            (self.df['ticker_count'] <= 3) &
            (self.df['content_length'] >= 200)
        )
        results.append(self.df[earnings_mask])

        # Product Launch / Regulatory: ticker in title OR extreme sentiment, <=2 tickers
        product_reg_mask = (
            (self.df['cat_Product Launch'] | self.df['cat_Regulatory/Legal']) &
            (
                (self.df['ticker_in_title']) |
                (self.df['sentiment_polarity'].abs() > 0.7)
            ) &
            (self.df['ticker_count'] <= 2) &
            (self.df['content_length'] >= 300)
        )
        results.append(self.df[product_reg_mask])

        # Analyst Ratings: ticker in title required, single ticker only
        analyst_mask = (
            (self.df['cat_Analyst Ratings']) &
            (self.df['ticker_in_title']) &
            (self.df['ticker_count'] == 1) &
            (self.df['content_length'] >= 150)
        )
        results.append(self.df[analyst_mask])

        # Executive Changes / Dividends: ticker in title, <=2 tickers
        exec_div_mask = (
            (self.df['cat_Executive Changes'] | self.df['cat_Dividends']) &
            (self.df['ticker_in_title']) &
            (self.df['ticker_count'] <= 2) &
            (self.df['content_length'] >= 200)
        )
        results.append(self.df[exec_div_mask])

        # Combine and deduplicate
        if results:
            filtered = pd.concat(results, ignore_index=True)
            filtered = filtered.drop_duplicates(subset=['date', 'title'], keep='first')
            return filtered
        else:
            return pd.DataFrame()

    def analyze_all_strategies(self):
        """Apply all strategies and collect statistics"""
        print(f"  → Applying all filtering strategies...")

        strategies = {
            'Original': self.df,
            'Precision': self.apply_strategy_precision(),
            'Recall': self.apply_strategy_recall(),
            'Balanced': self.apply_strategy_balanced(),
            'Category-Specific': self.apply_strategy_category_specific()
        }

        results = {}

        for name, filtered_df in strategies.items():
            if name == 'Original':
                stats = self.calculate_statistics(filtered_df, is_original=True)
            else:
                stats = self.calculate_statistics(filtered_df, is_original=False)

            results[name] = {
                'df': filtered_df,
                'stats': stats
            }

            print(f"    • {name:20s}: {len(filtered_df):6,} articles ({len(filtered_df)/len(self.df)*100:5.1f}%)")

        self.results = results
        return results

    def calculate_statistics(self, df, is_original=False):
        """Calculate comprehensive statistics for a filtered dataset"""
        if len(df) == 0:
            return {
                'article_count': 0,
                'unique_dates': 0,
                'avg_fp_score': np.nan,
                'ticker_in_title_pct': 0,
                'single_ticker_pct': 0,
                'avg_content_length': 0,
                'duplicate_pct': 0,
                'categorized_pct': 0,
                'category_distribution': {},
                'sentiment_mean': np.nan,
                'sentiment_std': np.nan,
                'extreme_sentiment_pct': 0
            }

        return {
            'article_count': len(df),
            'unique_dates': df['date'].dt.date.nunique(),
            'avg_fp_score': df['fp_score'].mean(),
            'ticker_in_title_pct': df['ticker_in_title'].sum() / len(df) * 100,
            'single_ticker_pct': (df['ticker_count'] == 1).sum() / len(df) * 100,
            'avg_content_length': df['content_length'].mean(),
            'duplicate_pct': df['is_duplicate'].sum() / len(df) * 100,
            'categorized_pct': (df['num_categories'] > 0).sum() / len(df) * 100,
            'category_distribution': df['primary_category'].value_counts().to_dict() if 'primary_category' in df.columns else {},
            'sentiment_mean': df['sentiment_polarity'].mean(),
            'sentiment_std': df['sentiment_polarity'].std(),
            'extreme_sentiment_pct': (df['sentiment_polarity'].abs() > 0.6).sum() / len(df) * 100
        }

    def save_filtered_datasets(self):
        """Save all filtered datasets"""
        print(f"  → Saving filtered datasets...")

        for strategy_name, data in self.results.items():
            if strategy_name == 'Original':
                continue

            df = data['df']
            if len(df) > 0:
                # Save full filtered dataset
                filename = f"{self.ticker}_{strategy_name.lower().replace(' ', '_')}_filtered.csv"
                filepath = OUTPUT_DIR / filename
                df.to_csv(filepath, index=False)

                # Save event dates only (for event studies)
                event_dates = pd.DataFrame({
                    'Date': df['date'].dt.date.unique()
                })
                event_dates = event_dates.sort_values('Date')
                dates_filename = f"{self.ticker}_{strategy_name.lower().replace(' ', '_')}_event_dates.csv"
                dates_filepath = OUTPUT_DIR / dates_filename
                event_dates.to_csv(dates_filepath, index=False)

        print(f"    ✓ Saved filtered datasets for {self.ticker}")

    def process(self):
        """Run complete filtering analysis"""
        print(f"\n{'='*80}")
        print(f"Processing {self.ticker} ({self.sector})")
        print(f"{'='*80}")

        if not self.load_data():
            return None

        self.detect_false_positives()
        self.categorize_events()
        self.analyze_all_strategies()
        self.save_filtered_datasets()

        return self.results


class FilteringComparison:
    """Compare filtering strategies across all stocks"""

    def __init__(self):
        self.all_results = {}

    def process_all_stocks(self):
        """Process all 50 stocks"""
        print("\n" + "="*80)
        print("COMPREHENSIVE NEWS FILTERING - ALL 50 STOCKS")
        print("="*80)

        for ticker, info in EXPANDED_STOCKS.items():
            filter_tool = ComprehensiveNewsFilter(
                ticker=ticker,
                sector=info['sector'],
                company_name=info['name']
            )

            results = filter_tool.process()
            if results:
                self.all_results[ticker] = results

        print(f"\n✓ Processed {len(self.all_results)} stocks successfully")

    def create_comparison_summary(self):
        """Create summary statistics comparing all strategies"""
        print(f"\n{'='*80}")
        print("Creating comparison summary...")
        print(f"{'='*80}")

        # Collect data for each strategy
        strategies = ['Original', 'Precision', 'Recall', 'Balanced', 'Category-Specific']

        summary_data = []

        for ticker, results in self.all_results.items():
            row = {'Ticker': ticker, 'Sector': EXPANDED_STOCKS[ticker]['sector']}

            for strategy in strategies:
                if strategy in results:
                    stats = results[strategy]['stats']
                    row[f'{strategy}_Count'] = stats['article_count']
                    row[f'{strategy}_Dates'] = stats['unique_dates']
                    row[f'{strategy}_FP_Score'] = stats['avg_fp_score']
                    row[f'{strategy}_TickerInTitle'] = stats['ticker_in_title_pct']
                    row[f'{strategy}_Sentiment'] = stats['sentiment_mean']

            summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)

        # Save summary
        summary_file = OUTPUT_DIR / "filtering_comparison_summary.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"✓ Saved comparison summary: {summary_file.name}")

        return df_summary

    def create_visualizations(self):
        """Create comprehensive comparison visualizations"""
        print(f"\n{'='*80}")
        print("Creating comparison visualizations...")
        print(f"{'='*80}")

        strategies = ['Precision', 'Recall', 'Balanced', 'Category-Specific']

        # Aggregate statistics across all stocks
        strategy_stats = {strategy: {
            'total_articles': 0,
            'avg_fp_score': [],
            'ticker_in_title_pct': [],
            'retention_pct': []
        } for strategy in strategies}

        for ticker, results in self.all_results.items():
            original_count = results['Original']['stats']['article_count']

            for strategy in strategies:
                if strategy in results:
                    stats = results[strategy]['stats']
                    strategy_stats[strategy]['total_articles'] += stats['article_count']
                    strategy_stats[strategy]['avg_fp_score'].append(stats['avg_fp_score'])
                    strategy_stats[strategy]['ticker_in_title_pct'].append(stats['ticker_in_title_pct'])
                    if original_count > 0:
                        strategy_stats[strategy]['retention_pct'].append(
                            stats['article_count'] / original_count * 100
                        )

        # Create figure with better spacing - wider to accommodate legend
        fig = plt.figure(figsize=(26, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.40)

        # 1. Total articles retained per strategy
        ax1 = fig.add_subplot(gs[0, 0])
        strategy_names = list(strategy_stats.keys())
        article_counts = [strategy_stats[s]['total_articles'] for s in strategy_names]
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        bars = ax1.bar(strategy_names, article_counts, color=colors, edgecolor='black', alpha=0.8)
        ax1.set_title('Total Articles Retained (All 50 Stocks)', fontsize=11, fontweight='bold', pad=15)
        ax1.set_ylabel('Total Articles', fontsize=10)
        ax1.tick_params(axis='x', rotation=38, labelsize=9, pad=8)
        # Fix data labels going outside - place inside bars for tall ones
        for i, (bar, count) in enumerate(zip(bars, article_counts)):
            if count > 80000:  # For tall bars, put label inside
                ax1.text(i, count * 0.95, f'{count:,}', ha='center', va='top', fontweight='bold', fontsize=9)
            else:
                ax1.text(i, count, f'\n{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax1.set_ylim([0, max(article_counts) * 1.08])  # Add headroom for labels
        ax1.grid(axis='y', alpha=0.3)

        # 2. Average retention percentage
        ax2 = fig.add_subplot(gs[0, 1])
        avg_retention = [np.mean(strategy_stats[s]['retention_pct']) for s in strategy_names]
        bars = ax2.bar(strategy_names, avg_retention, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_title('Average Retention Rate', fontsize=11, fontweight='bold', pad=15)
        ax2.set_ylabel('% of Original Articles', fontsize=10)
        ax2.tick_params(axis='x', rotation=38, labelsize=9, pad=8)
        # Fix labels - keep them inside plot area
        for i, (bar, pct) in enumerate(zip(bars, avg_retention)):
            ax2.text(i, pct + 1.5, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, max(avg_retention) * 1.18])  # Add more headroom

        # 3. Average false positive score
        ax3 = fig.add_subplot(gs[0, 2])
        avg_fp_scores = [np.nanmean(strategy_stats[s]['avg_fp_score']) for s in strategy_names]
        bars = ax3.bar(strategy_names, avg_fp_scores, color=colors, edgecolor='black', alpha=0.8)
        ax3.set_title('Average False Positive Score', fontsize=11, fontweight='bold', pad=15)
        ax3.set_ylabel('FP Score (0=Best, 3=Worst)', fontsize=10)
        ax3.tick_params(axis='x', rotation=38, labelsize=9, pad=8)
        # Fix labels
        for i, (bar, score) in enumerate(zip(bars, avg_fp_scores)):
            ax3.text(i, score + 0.08, f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, 3.2])  # Add headroom

        # 4. Ticker in title percentage
        ax4 = fig.add_subplot(gs[1, 0])
        avg_ticker_in_title = [np.mean(strategy_stats[s]['ticker_in_title_pct']) for s in strategy_names]
        bars = ax4.bar(strategy_names, avg_ticker_in_title, color=colors, edgecolor='black', alpha=0.8)
        ax4.set_title('Ticker in Title %', fontsize=11, fontweight='bold', pad=15)
        ax4.set_ylabel('% of Articles', fontsize=10)
        ax4.tick_params(axis='x', rotation=38, labelsize=9, pad=8)
        # Fix labels - place inside for 100%
        for i, (bar, pct) in enumerate(zip(bars, avg_ticker_in_title)):
            if pct > 95:  # For bars near 100%, put label inside
                ax4.text(i, pct - 3, f'{pct:.1f}%', ha='center', va='top', fontweight='bold', fontsize=9)
            else:
                ax4.text(i, pct + 2, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax4.set_ylim([0, 108])  # Add headroom
        ax4.grid(axis='y', alpha=0.3)

        # 5. Articles per stock distribution
        ax5 = fig.add_subplot(gs[1, 1])
        articles_per_stock = {strategy: [] for strategy in strategy_names}
        for ticker, results in self.all_results.items():
            for strategy in strategy_names:
                if strategy in results:
                    articles_per_stock[strategy].append(results[strategy]['stats']['article_count'])

        positions = range(len(strategy_names))
        bp = ax5.boxplot([articles_per_stock[s] for s in strategy_names],
                         labels=strategy_names,
                         patch_artist=True,
                         showfliers=False)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_title('Articles Per Stock Distribution', fontsize=11, fontweight='bold', pad=15)
        ax5.set_ylabel('Articles Per Stock', fontsize=10)
        ax5.tick_params(axis='x', rotation=38, labelsize=9, pad=8)
        ax5.grid(axis='y', alpha=0.3)

        # 6. Category distribution comparison
        ax6 = fig.add_subplot(gs[1, 2])
        # Aggregate category counts across all stocks for Balanced strategy
        balanced_categories = {}
        for ticker, results in self.all_results.items():
            if 'Balanced' in results:
                cat_dist = results['Balanced']['stats'].get('category_distribution', {})
                for cat, count in cat_dist.items():
                    balanced_categories[cat] = balanced_categories.get(cat, 0) + count

        if balanced_categories:
            sorted_cats = sorted(balanced_categories.items(), key=lambda x: x[1], reverse=True)[:8]
            cats, counts = zip(*sorted_cats)
            bars = ax6.barh(cats, counts, color='#2ecc71', edgecolor='black', alpha=0.8)
            ax6.set_title('Event Categories (Balanced Strategy)', fontsize=11, fontweight='bold', pad=15)
            ax6.set_xlabel('Total Articles', fontsize=10)
            # Fix labels - place them intelligently
            max_count = max(counts)
            for i, (bar, count) in enumerate(zip(bars, counts)):
                if count > max_count * 0.7:  # Long bars - put label inside
                    ax6.text(count * 0.95, i, f'{count:,}', va='center', ha='right', fontweight='bold', fontsize=9, color='white')
                else:
                    ax6.text(count + max_count * 0.01, i, f' {count:,}', va='center', ha='left', fontweight='bold', fontsize=9)
            ax6.set_xlim([0, max_count * 1.08])  # Add headroom

        # 7. Retention rate by stock (scatter) - WITH STOCK NAMES
        ax7 = fig.add_subplot(gs[2, :2])

        # Get stock tickers in order
        stock_tickers = list(self.all_results.keys())

        for strategy_idx, strategy in enumerate(strategy_names):
            retention_rates = []
            stock_positions = []

            for stock_idx, (ticker, results) in enumerate(self.all_results.items()):
                if strategy in results:
                    original = results['Original']['stats']['article_count']
                    filtered = results[strategy]['stats']['article_count']
                    if original > 0:
                        retention_rates.append(filtered / original * 100)
                        stock_positions.append(stock_idx)

            ax7.scatter(stock_positions, retention_rates, label=strategy,
                       color=colors[strategy_idx], alpha=0.6, s=50)

        ax7.set_title('Retention Rate by Stock', fontsize=11, fontweight='bold', pad=15)
        ax7.set_xlabel('Stock Ticker', fontsize=10, labelpad=10)
        ax7.set_ylabel('Retention %', fontsize=10)

        # Set x-axis with stock tickers
        ax7.set_xticks(range(len(stock_tickers)))
        ax7.set_xticklabels(stock_tickers, rotation=90, fontsize=7, ha='center')

        # Calculate actual data range to set appropriate limits
        all_retention_rates = []
        for strategy_idx, strategy in enumerate(strategy_names):
            for stock_idx, (ticker, results) in enumerate(self.all_results.items()):
                if strategy in results:
                    original = results['Original']['stats']['article_count']
                    filtered = results[strategy]['stats']['article_count']
                    if original > 0:
                        all_retention_rates.append(filtered / original * 100)

        max_retention = max(all_retention_rates) if all_retention_rates else 70

        # Set limits with headroom
        ax7.set_xlim([-1, len(stock_tickers)])
        ax7.set_ylim([-2, max_retention + 8])  # Add 8% headroom above max data point

        # Move legend outside plot area (to the right)
        ax7.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=True, fancybox=True)
        ax7.grid(alpha=0.3)

        # 8. Recommendation summary
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')

        recommendation_text = f"""STRATEGY RECOMMENDATIONS

Precision:
 • Clean event studies
 • Retention: {avg_retention[0]:.1f}%
 • FP Score: {avg_fp_scores[0]:.2f}
 • Quality > quantity

Recall:
 • Comprehensive analysis
 • Retention: {avg_retention[1]:.1f}%
 • FP Score: {avg_fp_scores[1]:.2f}
 • Don't miss events

Balanced (RECOMMENDED):
 • Most event studies
 • Retention: {avg_retention[2]:.1f}%
 • FP Score: {avg_fp_scores[2]:.2f}
 • Good trade-off

Category-Specific:
 • Robust research
 • Retention: {avg_retention[3]:.1f}%
 • FP Score: {avg_fp_scores[3]:.2f}
 • Different event types
"""

        ax8.text(0.05, 0.5, recommendation_text, fontsize=8.5, family='monospace',
                verticalalignment='center', linespacing=1.4)

        plt.suptitle('Comprehensive Filtering Strategy Comparison - 50 Stocks',
                     fontsize=16, fontweight='bold', y=0.997)

        # Adjust layout with padding - more space at bottom for stock names
        plt.tight_layout(rect=[0, 0.03, 1, 0.985], pad=3.0)

        # Save figure
        output_file = OUTPUT_DIR / 'filtering_strategies_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"✓ Saved visualization: {output_file.name}")
        plt.close()

    def generate_recommendations(self):
        """Generate stock-specific recommendations"""
        print(f"\n{'='*80}")
        print("Generating stock-specific recommendations...")
        print(f"{'='*80}")

        recommendations = []

        for ticker, results in self.all_results.items():
            original_count = results['Original']['stats']['article_count']

            # Analyze each strategy's suitability
            strategy_scores = {}

            for strategy in ['Precision', 'Recall', 'Balanced', 'Category-Specific']:
                if strategy in results:
                    stats = results[strategy]['stats']

                    # Score based on multiple factors
                    score = 0

                    # Sufficient data (25+ articles)
                    if stats['article_count'] >= 25:
                        score += 2
                    elif stats['article_count'] >= 10:
                        score += 1

                    # Low FP score
                    if stats['avg_fp_score'] < 0.5:
                        score += 2
                    elif stats['avg_fp_score'] < 1.0:
                        score += 1

                    # Good ticker in title percentage
                    if stats['ticker_in_title_pct'] > 80:
                        score += 2
                    elif stats['ticker_in_title_pct'] > 50:
                        score += 1

                    # Reasonable retention (not too high, not too low)
                    retention = stats['article_count'] / original_count * 100 if original_count > 0 else 0
                    if 15 <= retention <= 40:
                        score += 2
                    elif 10 <= retention <= 60:
                        score += 1

                    strategy_scores[strategy] = score

            # Recommend best strategy
            if strategy_scores:
                best_strategy = max(strategy_scores, key=strategy_scores.get)
                best_score = strategy_scores[best_strategy]
            else:
                best_strategy = 'N/A'
                best_score = 0

            recommendations.append({
                'Ticker': ticker,
                'Sector': EXPANDED_STOCKS[ticker]['sector'],
                'Original_Articles': original_count,
                'Recommended_Strategy': best_strategy,
                'Confidence_Score': best_score,
                'Precision_Articles': results.get('Precision', {}).get('stats', {}).get('article_count', 0),
                'Recall_Articles': results.get('Recall', {}).get('stats', {}).get('article_count', 0),
                'Balanced_Articles': results.get('Balanced', {}).get('stats', {}).get('article_count', 0),
                'CategorySpecific_Articles': results.get('Category-Specific', {}).get('stats', {}).get('article_count', 0)
            })

        df_recommendations = pd.DataFrame(recommendations)

        # Save recommendations
        rec_file = OUTPUT_DIR / "strategy_recommendations.csv"
        df_recommendations.to_csv(rec_file, index=False)
        print(f"✓ Saved recommendations: {rec_file.name}")

        # Print summary
        print(f"\n📊 Recommendation Summary:")
        rec_dist = df_recommendations['Recommended_Strategy'].value_counts()
        for strategy, count in rec_dist.items():
            print(f"  • {strategy:20s}: {count:2d} stocks ({count/len(df_recommendations)*100:5.1f}%)")

        return df_recommendations


def main():
    """Run comprehensive filtering analysis"""
    print("\n" + "="*80)
    print("COMPREHENSIVE NEWS FILTERING SYSTEM")
    print("Side-by-Side Comparison of 4 Filtering Strategies")
    print("="*80)
    print(f"\nStrategies:")
    print(f"  1. Precision    - Low false positives, highest quality")
    print(f"  2. Recall       - Comprehensive coverage, more inclusive")
    print(f"  3. Balanced     - Trade-off between quality and coverage")
    print(f"  4. Category-Specific - Different rules per event type")
    print("="*80)

    # Create comparison tool
    comparison = FilteringComparison()

    # Process all stocks
    comparison.process_all_stocks()

    # Create summary and visualizations
    comparison.create_comparison_summary()
    comparison.create_visualizations()
    comparison.generate_recommendations()

    print("\n" + "="*80)
    print("✅ COMPREHENSIVE FILTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print(f"\nGenerated Files:")
    print(f"  • filtering_comparison_summary.csv - Detailed statistics for all strategies")
    print(f"  • filtering_strategies_comparison.png - Visual comparison dashboard")
    print(f"  • strategy_recommendations.csv - Stock-specific recommendations")
    print(f"  • Individual filtered datasets for each stock × strategy")
    print(f"  • Event date files for each stock × strategy")
    print("="*80)


if __name__ == "__main__":
    main()
