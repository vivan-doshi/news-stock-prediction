"""
COMPREHENSIVE NEWS EDA
======================

Comprehensive exploratory data analysis of news data across all 50 stocks.

This script answers key questions about the news data:
1. Volume & Frequency patterns
2. Content quality assessment
3. Event categorization
4. Competitor & cross-stock analysis
5. Sentiment deep dive
6. Source diversity
7. False positive/negative detection
8. Temporal patterns
9. Data quality issues
10. Event study readiness

Author: Claude
Date: 2025-10-12
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import warnings
import re
from urllib.parse import urlparse
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

# Import stock configuration
config = import_module('21_expanded_50_stock_config')
EXPANDED_STOCKS = config.EXPANDED_STOCKS

# Directories
DATA_DIR = Path("../01-data")
OUTPUT_DIR = Path("../03-output/news_eda")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ComprehensiveNewsEDA:
    """Comprehensive EDA for news data across all stocks"""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.stocks = EXPANDED_STOCKS
        self.all_news = None
        self.results = {}

    def load_all_news(self):
        """Load news data for all 50 stocks"""
        print("="*80)
        print("LOADING NEWS DATA FOR ALL STOCKS")
        print("="*80)

        all_dfs = []
        loaded_stocks = []
        missing_stocks = []

        for ticker in self.stocks.keys():
            news_file = self.data_dir / f"{ticker}_eodhd_news.csv"

            if news_file.exists():
                try:
                    df = pd.read_csv(news_file)

                    # Add metadata
                    df['sector'] = self.stocks[ticker]['sector']
                    df['company_name'] = self.stocks[ticker]['name']

                    all_dfs.append(df)
                    loaded_stocks.append(ticker)
                    print(f"✅ {ticker:6s} - {len(df):6,} articles - {self.stocks[ticker]['name']}")

                except Exception as e:
                    print(f"❌ {ticker:6s} - Error loading: {e}")
                    missing_stocks.append(ticker)
            else:
                print(f"⚠️  {ticker:6s} - File not found")
                missing_stocks.append(ticker)

        if all_dfs:
            self.all_news = pd.concat(all_dfs, ignore_index=True)

            # Parse dates
            self.all_news['date'] = pd.to_datetime(self.all_news['date'], utc=True, errors='coerce')
            self.all_news['date_only'] = self.all_news['date'].dt.date
            self.all_news['year'] = self.all_news['date'].dt.year
            self.all_news['month'] = self.all_news['date'].dt.month
            self.all_news['day_of_week'] = self.all_news['date'].dt.dayofweek
            self.all_news['hour'] = self.all_news['date'].dt.hour

            print(f"\n{'='*80}")
            print(f"📊 SUMMARY")
            print(f"{'='*80}")
            print(f"✅ Loaded stocks: {len(loaded_stocks)}/50")
            print(f"❌ Missing stocks: {len(missing_stocks)}")
            print(f"📰 Total articles: {len(self.all_news):,}")
            print(f"📅 Date range: {self.all_news['date'].min()} to {self.all_news['date'].max()}")

            if missing_stocks:
                print(f"\nMissing: {', '.join(missing_stocks)}")
        else:
            raise ValueError("No news data loaded!")

        return self.all_news

    # ====================================================================
    # MODULE 1: VOLUME & FREQUENCY ANALYSIS
    # ====================================================================

    def analyze_volume_frequency(self):
        """Analyze news volume and frequency patterns"""
        print(f"\n{'='*80}")
        print("MODULE 1: VOLUME & FREQUENCY ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Overall statistics
        results['total_articles'] = len(self.all_news)
        results['total_stocks'] = self.all_news['ticker'].nunique()
        results['date_range_days'] = (self.all_news['date'].max() - self.all_news['date'].min()).days

        # Articles per stock
        articles_per_stock = self.all_news.groupby('ticker').size().sort_values(ascending=False)
        results['articles_per_stock'] = articles_per_stock.to_dict()

        print(f"\n📊 OVERALL STATISTICS:")
        print(f"  Total articles: {results['total_articles']:,}")
        print(f"  Unique stocks: {results['total_stocks']}")
        print(f"  Date range: {results['date_range_days']} days")
        print(f"  Avg articles per stock: {results['total_articles']/results['total_stocks']:.0f}")

        print(f"\n📈 TOP 10 STOCKS BY ARTICLE COUNT:")
        for ticker, count in articles_per_stock.head(10).items():
            name = self.stocks[ticker]['name']
            print(f"  {ticker:6s} ({name:20s}): {count:6,} articles")

        print(f"\n📉 BOTTOM 10 STOCKS BY ARTICLE COUNT:")
        for ticker, count in articles_per_stock.tail(10).items():
            name = self.stocks[ticker]['name']
            print(f"  {ticker:6s} ({name:20s}): {count:6,} articles")

        # Articles per sector
        articles_per_sector = self.all_news.groupby('sector').size().sort_values(ascending=False)
        results['articles_per_sector'] = articles_per_sector.to_dict()

        print(f"\n🏢 ARTICLES BY SECTOR:")
        for sector, count in articles_per_sector.items():
            print(f"  {sector:30s}: {count:7,} articles")

        # Daily frequency
        daily_counts = self.all_news.groupby('date_only').size()
        results['avg_articles_per_day'] = daily_counts.mean()
        results['median_articles_per_day'] = daily_counts.median()
        results['max_articles_in_day'] = daily_counts.max()
        results['days_with_news'] = len(daily_counts)

        print(f"\n📅 DAILY FREQUENCY:")
        print(f"  Average articles per day: {results['avg_articles_per_day']:.1f}")
        print(f"  Median articles per day: {results['median_articles_per_day']:.1f}")
        print(f"  Max articles in one day: {results['max_articles_in_day']}")
        print(f"  Days with news: {results['days_with_news']}")

        # High volume days
        top_days = daily_counts.nlargest(20)
        print(f"\n🔥 TOP 20 HIGH-VOLUME NEWS DAYS:")
        for date, count in top_days.items():
            print(f"  {date}: {count} articles")

        # Create visualizations
        self._plot_volume_analysis(articles_per_stock, articles_per_sector, daily_counts)

        self.results['volume_frequency'] = results
        return results

    def _plot_volume_analysis(self, articles_per_stock, articles_per_sector, daily_counts):
        """Create volume analysis visualizations"""

        fig = plt.figure(figsize=(20, 12))

        # 1. Articles per stock (top 20)
        ax1 = plt.subplot(3, 3, 1)
        top_20 = articles_per_stock.head(20)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20)))
        top_20.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Top 20 Stocks by Article Count', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Number of Articles')
        ax1.invert_yaxis()

        # 2. Articles per sector
        ax2 = plt.subplot(3, 3, 2)
        articles_per_sector.plot(kind='bar', ax=ax2, color='steelblue', alpha=0.7)
        ax2.set_title('Articles by Sector', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Sector')
        ax2.set_ylabel('Number of Articles')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Articles distribution
        ax3 = plt.subplot(3, 3, 3)
        articles_per_stock.hist(bins=30, ax=ax3, edgecolor='black', alpha=0.7)
        ax3.set_title('Distribution of Articles per Stock', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Number of Articles')
        ax3.set_ylabel('Number of Stocks')
        ax3.axvline(articles_per_stock.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {articles_per_stock.mean():.0f}')
        ax3.axvline(articles_per_stock.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {articles_per_stock.median():.0f}')
        ax3.legend()

        # 4. Daily news volume over time
        ax4 = plt.subplot(3, 3, (4, 6))
        daily_counts.plot(ax=ax4, linewidth=0.8, alpha=0.7, color='darkblue')
        ax4.set_title('Daily News Volume Over Time (All Stocks)', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Number of Articles')
        ax4.grid(alpha=0.3)

        # 5. Daily volume distribution
        ax5 = plt.subplot(3, 3, 7)
        daily_counts.hist(bins=50, ax=ax5, edgecolor='black', alpha=0.7, color='coral')
        ax5.axvline(daily_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_counts.mean():.1f}')
        ax5.axvline(daily_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {daily_counts.median():.1f}')
        ax5.set_title('Distribution of Daily Article Count', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Articles per Day')
        ax5.set_ylabel('Frequency')
        ax5.legend()

        # 6. Monthly trend
        ax6 = plt.subplot(3, 3, 8)
        monthly_counts = self.all_news.groupby([self.all_news['date'].dt.to_period('M')]).size()
        monthly_counts.plot(ax=ax6, marker='o', linewidth=2, markersize=4)
        ax6.set_title('Monthly News Volume Trend', fontweight='bold', fontsize=12)
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Number of Articles')
        ax6.grid(alpha=0.3)
        ax6.tick_params(axis='x', rotation=45)

        # 7. Yearly comparison
        ax7 = plt.subplot(3, 3, 9)
        yearly_counts = self.all_news.groupby('year').size()
        yearly_counts.plot(kind='bar', ax=ax7, color='purple', alpha=0.7)
        ax7.set_title('Articles by Year', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Year')
        ax7.set_ylabel('Number of Articles')
        ax7.tick_params(axis='x', rotation=0)

        plt.tight_layout()
        plt.savefig(self.output_dir / '01_volume_frequency_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 01_volume_frequency_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 2: CONTENT QUALITY ANALYSIS
    # ====================================================================

    def analyze_content_quality(self):
        """Analyze content quality and completeness"""
        print(f"\n{'='*80}")
        print("MODULE 2: CONTENT QUALITY ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Missing values
        missing_stats = {}
        for col in self.all_news.columns:
            missing_count = self.all_news[col].isnull().sum()
            missing_pct = (missing_count / len(self.all_news)) * 100
            missing_stats[col] = {'count': missing_count, 'percentage': missing_pct}

        results['missing_values'] = missing_stats

        print(f"\n❓ MISSING VALUES:")
        for col, stats in missing_stats.items():
            if stats['count'] > 0:
                print(f"  {col:25s}: {stats['count']:8,} ({stats['percentage']:5.2f}%)")

        # Title analysis
        self.all_news['title_length'] = self.all_news['title'].astype(str).str.len()
        self.all_news['title_words'] = self.all_news['title'].astype(str).str.split().str.len()

        results['title_length_mean'] = self.all_news['title_length'].mean()
        results['title_length_median'] = self.all_news['title_length'].median()
        results['title_words_mean'] = self.all_news['title_words'].mean()
        results['title_words_median'] = self.all_news['title_words'].median()

        print(f"\n📏 TITLE STATISTICS:")
        print(f"  Average characters: {results['title_length_mean']:.1f}")
        print(f"  Median characters: {results['title_length_median']:.1f}")
        print(f"  Average words: {results['title_words_mean']:.1f}")
        print(f"  Median words: {results['title_words_median']:.1f}")

        # Content analysis
        self.all_news['content_length'] = self.all_news['content'].astype(str).str.len()
        self.all_news['content_words'] = self.all_news['content'].astype(str).str.split().str.len()

        results['content_length_mean'] = self.all_news['content_length'].mean()
        results['content_length_median'] = self.all_news['content_length'].median()
        results['content_words_mean'] = self.all_news['content_words'].mean()
        results['content_words_median'] = self.all_news['content_words'].median()

        print(f"\n📄 CONTENT STATISTICS:")
        print(f"  Average characters: {results['content_length_mean']:.1f}")
        print(f"  Median characters: {results['content_length_median']:.1f}")
        print(f"  Average words: {results['content_words_mean']:.1f}")
        print(f"  Median words: {results['content_words_median']:.1f}")

        # Empty or very short content
        empty_content = (self.all_news['content_length'] < 50).sum()
        empty_pct = (empty_content / len(self.all_news)) * 100
        results['empty_content_count'] = empty_content
        results['empty_content_pct'] = empty_pct

        print(f"\n⚠️  QUALITY ISSUES:")
        print(f"  Very short content (<50 chars): {empty_content:,} ({empty_pct:.2f}%)")

        # Duplicate detection (same title)
        duplicates = self.all_news.duplicated(subset=['title'], keep=False).sum()
        duplicate_pct = (duplicates / len(self.all_news)) * 100
        results['duplicate_titles'] = duplicates
        results['duplicate_titles_pct'] = duplicate_pct

        print(f"  Duplicate titles: {duplicates:,} ({duplicate_pct:.2f}%)")

        # Create visualizations
        self._plot_content_quality()

        self.results['content_quality'] = results
        return results

    def _plot_content_quality(self):
        """Create content quality visualizations"""

        fig = plt.figure(figsize=(16, 10))

        # 1. Title length distribution
        ax1 = plt.subplot(2, 3, 1)
        self.all_news['title_length'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(self.all_news['title_length'].mean(), color='red', linestyle='--', linewidth=2)
        ax1.set_title('Title Length Distribution (Characters)', fontweight='bold')
        ax1.set_xlabel('Characters')
        ax1.set_ylabel('Frequency')

        # 2. Title words distribution
        ax2 = plt.subplot(2, 3, 2)
        self.all_news['title_words'].hist(bins=30, ax=ax2, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(self.all_news['title_words'].mean(), color='red', linestyle='--', linewidth=2)
        ax2.set_title('Title Length Distribution (Words)', fontweight='bold')
        ax2.set_xlabel('Words')
        ax2.set_ylabel('Frequency')

        # 3. Content length distribution
        ax3 = plt.subplot(2, 3, 3)
        # Filter outliers for better visualization
        content_filtered = self.all_news['content_length'][self.all_news['content_length'] < 5000]
        content_filtered.hist(bins=50, ax=ax3, edgecolor='black', alpha=0.7, color='green')
        ax3.axvline(self.all_news['content_length'].mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_title('Content Length Distribution (Characters, <5000)', fontweight='bold')
        ax3.set_xlabel('Characters')
        ax3.set_ylabel('Frequency')

        # 4. Content words distribution
        ax4 = plt.subplot(2, 3, 4)
        words_filtered = self.all_news['content_words'][self.all_news['content_words'] < 1000]
        words_filtered.hist(bins=50, ax=ax4, edgecolor='black', alpha=0.7, color='purple')
        ax4.axvline(self.all_news['content_words'].mean(), color='red', linestyle='--', linewidth=2)
        ax4.set_title('Content Length Distribution (Words, <1000)', fontweight='bold')
        ax4.set_xlabel('Words')
        ax4.set_ylabel('Frequency')

        # 5. Missing values heatmap
        ax5 = plt.subplot(2, 3, 5)
        missing_data = self.all_news.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            missing_data.plot(kind='barh', ax=ax5, color='red', alpha=0.7)
            ax5.set_title('Missing Values by Column', fontweight='bold')
            ax5.set_xlabel('Count')
        else:
            ax5.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax5.set_title('Missing Values by Column', fontweight='bold')

        # 6. Quality score (composite)
        ax6 = plt.subplot(2, 3, 6)
        # Create a simple quality score
        quality_score = (
            (self.all_news['title_length'] > 10).astype(int) +
            (self.all_news['content_length'] > 100).astype(int) +
            (~self.all_news['content'].isnull()).astype(int) +
            (~self.all_news['link'].isnull()).astype(int)
        ) / 4 * 100

        quality_score.hist(bins=20, ax=ax6, edgecolor='black', alpha=0.7, color='gold')
        ax6.set_title('Content Quality Score Distribution', fontweight='bold')
        ax6.set_xlabel('Quality Score (%)')
        ax6.set_ylabel('Frequency')
        ax6.axvline(quality_score.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {quality_score.mean():.1f}%')
        ax6.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / '02_content_quality_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 02_content_quality_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 3: EVENT CLASSIFICATION
    # ====================================================================

    def analyze_event_categories(self):
        """Classify news into event categories"""
        print(f"\n{'='*80}")
        print("MODULE 3: EVENT CLASSIFICATION")
        print(f"{'='*80}")

        # Define event categories with keywords
        event_categories = {
            'Earnings': ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4',
                        'revenue', 'profit', 'eps', 'guidance', 'fiscal'],
            'Product Launch': ['launch', 'release', 'unveil', 'introduce', 'new product', 'new model',
                              'announce', 'debut', 'rollout'],
            'Executive Changes': ['ceo', 'executive', 'leadership', 'management', 'board', 'resign',
                                 'appoint', 'hire', 'retire', 'chief', 'director'],
            'M&A': ['merger', 'acquisition', 'acquire', 'deal', 'takeover', 'buy', 'purchase',
                   'stake', 'partnership', 'joint venture'],
            'Regulatory/Legal': ['sec', 'lawsuit', 'investigation', 'probe', 'regulator', 'compliance',
                                'legal', 'court', 'fine', 'penalty', 'litigation'],
            'Market Performance': ['stock', 'shares', 'market', 'trading', 'price', 'rally', 'drop',
                                  'surge', 'plunge', 'soar', 'tumble', 'gain', 'loss'],
            'Operations': ['layoff', 'restructure', 'close', 'shutdown', 'expansion', 'plant',
                          'facility', 'production', 'manufacturing'],
            'Technology/Innovation': ['patent', 'innovation', 'technology', 'r&d', 'research',
                                     'development', 'breakthrough', 'ai', 'software'],
            'Analyst Ratings': ['upgrade', 'downgrade', 'analyst', 'rating', 'price target',
                               'recommendation', 'outperform', 'underperform'],
            'Dividends': ['dividend', 'payout', 'yield', 'distribution', 'shareholder return']
        }

        # Classify articles
        for category, keywords in event_categories.items():
            pattern = '|'.join(keywords)
            self.all_news[f'cat_{category}'] = self.all_news['title'].str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)

        # Count articles per category
        category_counts = {}
        for category in event_categories.keys():
            count = self.all_news[f'cat_{category}'].sum()
            category_counts[category] = count
            percentage = (count / len(self.all_news)) * 100
            print(f"\n📌 {category.upper()}: {count:,} articles ({percentage:.2f}%)")

            # Show examples
            examples = self.all_news[self.all_news[f'cat_{category}'] == 1].head(5)
            if len(examples) > 0:
                print(f"  Examples:")
                for idx, row in examples.iterrows():
                    date = row['date_only']
                    ticker = row['ticker']
                    title = row['title'][:70]
                    print(f"    • [{ticker}] {date}: {title}...")

        # Articles with multiple categories
        category_cols = [f'cat_{cat}' for cat in event_categories.keys()]
        self.all_news['total_categories'] = self.all_news[category_cols].sum(axis=1)

        multi_cat = (self.all_news['total_categories'] > 1).sum()
        multi_cat_pct = (multi_cat / len(self.all_news)) * 100

        print(f"\n🔀 MULTI-CATEGORY ARTICLES:")
        print(f"  Articles in multiple categories: {multi_cat:,} ({multi_cat_pct:.2f}%)")

        # Uncategorized articles
        uncategorized = (self.all_news['total_categories'] == 0).sum()
        uncategorized_pct = (uncategorized / len(self.all_news)) * 100

        print(f"\n❓ UNCATEGORIZED ARTICLES:")
        print(f"  Articles not matching any category: {uncategorized:,} ({uncategorized_pct:.2f}%)")

        # Create visualizations
        self._plot_event_categories(category_counts, event_categories)

        self.results['event_categories'] = {
            'category_counts': category_counts,
            'multi_category_count': multi_cat,
            'uncategorized_count': uncategorized
        }

        return category_counts

    def _plot_event_categories(self, category_counts, event_categories):
        """Create event category visualizations"""

        fig = plt.figure(figsize=(18, 10))

        # 1. Overall category counts
        ax1 = plt.subplot(2, 3, 1)
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

        bars = ax1.barh(categories, counts, color=colors, alpha=0.8)
        ax1.set_title('Articles by Event Category', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Number of Articles')

        # Add value labels
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax1.text(count + max(counts)*0.01, i, f'{count:,}', va='center')

        # 2. Category distribution pie chart
        ax2 = plt.subplot(2, 3, 2)
        top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:7])
        others_count = sum([v for k, v in category_counts.items() if k not in top_categories])
        if others_count > 0:
            top_categories['Others'] = others_count

        ax2.pie(top_categories.values(), labels=top_categories.keys(), autopct='%1.1f%%', startangle=90)
        ax2.set_title('Category Distribution (Top 7 + Others)', fontweight='bold', fontsize=12)

        # 3. Categories by sector
        ax3 = plt.subplot(2, 3, 3)
        category_cols = [f'cat_{cat}' for cat in event_categories.keys()]
        sector_category = self.all_news.groupby('sector')[category_cols].sum().sum(axis=1).sort_values(ascending=False)
        sector_category.plot(kind='barh', ax=ax3, color='steelblue', alpha=0.7)
        ax3.set_title('Categorized Articles by Sector', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Number of Articles')

        # 4. Categories over time
        ax4 = plt.subplot(2, 3, (4, 6))
        # Get monthly counts for top 5 categories
        top_5_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        for cat_name, _ in top_5_cats:
            monthly = self.all_news[self.all_news[f'cat_{cat_name}'] == 1].groupby(
                self.all_news['date'].dt.to_period('M')
            ).size()
            monthly.plot(ax=ax4, label=cat_name, linewidth=2, alpha=0.7)

        ax4.set_title('Top 5 Event Categories Over Time (Monthly)', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Articles')
        ax4.legend(loc='best')
        ax4.grid(alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / '03_event_categories_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 03_event_categories_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 4: COMPETITOR & CROSS-STOCK ANALYSIS
    # ====================================================================

    def analyze_competitor_crossstock(self):
        """Analyze competitor mentions and cross-stock coverage"""
        print(f"\n{'='*80}")
        print("MODULE 4: COMPETITOR & CROSS-STOCK ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Parse symbols field
        def parse_symbols(symbols_str):
            if pd.isna(symbols_str):
                return []
            # Parse list-like strings
            symbols_str = str(symbols_str)
            symbols = re.findall(r'[A-Z]{1,5}\.US', symbols_str)
            return [s.replace('.US', '') for s in symbols]

        self.all_news['parsed_symbols'] = self.all_news['symbols'].apply(parse_symbols)
        self.all_news['num_symbols'] = self.all_news['parsed_symbols'].apply(len)

        # Multi-ticker articles
        multi_ticker = (self.all_news['num_symbols'] > 1).sum()
        multi_ticker_pct = (multi_ticker / len(self.all_news)) * 100

        results['multi_ticker_count'] = multi_ticker
        results['multi_ticker_pct'] = multi_ticker_pct

        print(f"\n🔗 MULTI-TICKER ARTICLES:")
        print(f"  Articles mentioning multiple tickers: {multi_ticker:,} ({multi_ticker_pct:.2f}%)")

        # Most common co-mentions
        print(f"\n🤝 TOP CO-MENTIONED STOCK PAIRS:")
        co_mentions = defaultdict(int)
        for symbols in self.all_news['parsed_symbols']:
            if len(symbols) > 1:
                for i, s1 in enumerate(symbols):
                    for s2 in symbols[i+1:]:
                        pair = tuple(sorted([s1, s2]))
                        co_mentions[pair] += 1

        top_pairs = sorted(co_mentions.items(), key=lambda x: x[1], reverse=True)[:15]
        for (s1, s2), count in top_pairs:
            print(f"  {s1} + {s2}: {count} articles")

        results['top_co_mentions'] = {f"{s1}+{s2}": count for (s1, s2), count in top_pairs}

        # Sector-wide news
        print(f"\n🏢 SECTOR-WIDE NEWS ANALYSIS:")
        for sector in self.all_news['sector'].unique():
            sector_stocks = [t for t, info in self.stocks.items() if info['sector'] == sector]

            # Count articles mentioning multiple stocks from same sector
            sector_wide = 0
            for idx, row in self.all_news[self.all_news['sector'] == sector].iterrows():
                symbols = row['parsed_symbols']
                if len([s for s in symbols if s in sector_stocks]) > 1:
                    sector_wide += 1

            total_sector = len(self.all_news[self.all_news['sector'] == sector])
            pct = (sector_wide / total_sector * 100) if total_sector > 0 else 0
            print(f"  {sector:30s}: {sector_wide:5,} multi-stock articles ({pct:.2f}%)")

        # Create visualizations
        self._plot_competitor_analysis(results, top_pairs)

        self.results['competitor_crossstock'] = results
        return results

    def _plot_competitor_analysis(self, results, top_pairs):
        """Create competitor/cross-stock visualizations"""

        fig = plt.figure(figsize=(16, 10))

        # 1. Distribution of number of symbols
        ax1 = plt.subplot(2, 2, 1)
        symbol_dist = self.all_news['num_symbols'].value_counts().sort_index()
        symbol_dist.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
        ax1.set_title('Distribution of Ticker Count per Article', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Number of Tickers Mentioned')
        ax1.set_ylabel('Number of Articles')
        ax1.tick_params(axis='x', rotation=0)

        # 2. Top co-mentioned pairs
        ax2 = plt.subplot(2, 2, 2)
        if top_pairs:
            pairs_labels = [f"{s1}+{s2}" for (s1, s2), _ in top_pairs[:15]]
            pairs_counts = [count for _, count in top_pairs[:15]]

            ax2.barh(pairs_labels, pairs_counts, color='coral', alpha=0.7)
            ax2.set_title('Top 15 Co-Mentioned Stock Pairs', fontweight='bold', fontsize=12)
            ax2.set_xlabel('Number of Articles')
            ax2.invert_yaxis()

        # 3. Multi-ticker percentage by sector
        ax3 = plt.subplot(2, 2, 3)
        sector_multi = self.all_news[self.all_news['num_symbols'] > 1].groupby('sector').size()
        sector_total = self.all_news.groupby('sector').size()
        sector_pct = (sector_multi / sector_total * 100).sort_values(ascending=False)

        sector_pct.plot(kind='barh', ax=ax3, color='green', alpha=0.7)
        ax3.set_title('Multi-Ticker Articles by Sector (%)', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Percentage')

        # 4. Average symbols per stock
        ax4 = plt.subplot(2, 2, 4)
        avg_symbols = self.all_news.groupby('ticker')['num_symbols'].mean().sort_values(ascending=False).head(20)
        avg_symbols.plot(kind='barh', ax=ax4, color='purple', alpha=0.7)
        ax4.set_title('Top 20 Stocks: Avg Tickers per Article', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Average Number of Tickers')
        ax4.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / '04_competitor_crossstock_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 04_competitor_crossstock_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 5: SENTIMENT ANALYSIS
    # ====================================================================

    def analyze_sentiment(self):
        """Deep dive into sentiment analysis"""
        print(f"\n{'='*80}")
        print("MODULE 5: SENTIMENT ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Check for sentiment columns
        sentiment_cols = ['sentiment_polarity', 'sentiment_neg', 'sentiment_neu', 'sentiment_pos']

        print(f"\n📊 SENTIMENT STATISTICS:")

        for col in sentiment_cols:
            if col in self.all_news.columns:
                results[f'{col}_mean'] = self.all_news[col].mean()
                results[f'{col}_median'] = self.all_news[col].median()
                results[f'{col}_std'] = self.all_news[col].std()

                print(f"\n  {col.upper()}:")
                print(f"    Mean: {results[f'{col}_mean']:.3f}")
                print(f"    Median: {results[f'{col}_median']:.3f}")
                print(f"    Std: {results[f'{col}_std']:.3f}")
                print(f"    Min: {self.all_news[col].min():.3f}")
                print(f"    Max: {self.all_news[col].max():.3f}")

        # Sentiment polarity categorization
        if 'sentiment_polarity' in self.all_news.columns:
            self.all_news['sentiment_category'] = pd.cut(
                self.all_news['sentiment_polarity'],
                bins=[-float('inf'), -0.5, -0.1, 0.1, 0.5, float('inf')],
                labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
            )

            sentiment_dist = self.all_news['sentiment_category'].value_counts()

            print(f"\n😊 SENTIMENT DISTRIBUTION:")
            for cat, count in sentiment_dist.items():
                pct = (count / len(self.all_news)) * 100
                print(f"  {cat:15s}: {count:7,} ({pct:5.2f}%)")

            results['sentiment_distribution'] = sentiment_dist.to_dict()

            # Sentiment by sector
            print(f"\n🏢 AVERAGE SENTIMENT BY SECTOR:")
            sector_sentiment = self.all_news.groupby('sector')['sentiment_polarity'].mean().sort_values(ascending=False)
            for sector, sent in sector_sentiment.items():
                print(f"  {sector:30s}: {sent:6.3f}")

            results['sector_sentiment'] = sector_sentiment.to_dict()

            # Sentiment by category
            print(f"\n📌 AVERAGE SENTIMENT BY EVENT CATEGORY:")
            category_cols = [c for c in self.all_news.columns if c.startswith('cat_')]
            for col in category_cols:
                cat_name = col.replace('cat_', '')
                cat_articles = self.all_news[self.all_news[col] == 1]
                if len(cat_articles) > 0:
                    avg_sent = cat_articles['sentiment_polarity'].mean()
                    print(f"  {cat_name:25s}: {avg_sent:6.3f}")

            # Extreme sentiment examples
            print(f"\n🔴 MOST NEGATIVE NEWS (Top 10):")
            most_negative = self.all_news.nsmallest(10, 'sentiment_polarity')
            for idx, row in most_negative.iterrows():
                print(f"  [{row['ticker']}] {row['date_only']}: {row['title'][:60]}... (sentiment: {row['sentiment_polarity']:.3f})")

            print(f"\n🟢 MOST POSITIVE NEWS (Top 10):")
            most_positive = self.all_news.nlargest(10, 'sentiment_polarity')
            for idx, row in most_positive.iterrows():
                print(f"  [{row['ticker']}] {row['date_only']}: {row['title'][:60]}... (sentiment: {row['sentiment_polarity']:.3f})")

        # Create visualizations
        self._plot_sentiment_analysis()

        self.results['sentiment'] = results
        return results

    def _plot_sentiment_analysis(self):
        """Create sentiment analysis visualizations"""

        if 'sentiment_polarity' not in self.all_news.columns:
            print("⚠️  No sentiment data available for visualization")
            return

        fig = plt.figure(figsize=(18, 12))

        # 1. Sentiment polarity distribution
        ax1 = plt.subplot(3, 3, 1)
        self.all_news['sentiment_polarity'].hist(bins=50, ax=ax1, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(self.all_news['sentiment_polarity'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_title('Sentiment Polarity Distribution', fontweight='bold')
        ax1.set_xlabel('Sentiment Polarity')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # 2. Sentiment categories
        ax2 = plt.subplot(3, 3, 2)
        if 'sentiment_category' in self.all_news.columns:
            sentiment_counts = self.all_news['sentiment_category'].value_counts()
            colors = ['darkred', 'salmon', 'gray', 'lightgreen', 'darkgreen']
            sentiment_counts.plot(kind='bar', ax=ax2, color=colors, alpha=0.7)
            ax2.set_title('Sentiment Categories', fontweight='bold')
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Number of Articles')
            ax2.tick_params(axis='x', rotation=45)

        # 3. Sentiment by sector
        ax3 = plt.subplot(3, 3, 3)
        sector_sentiment = self.all_news.groupby('sector')['sentiment_polarity'].mean().sort_values()
        colors = ['red' if x < 0 else 'green' for x in sector_sentiment]
        sector_sentiment.plot(kind='barh', ax=ax3, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Average Sentiment by Sector', fontweight='bold')
        ax3.set_xlabel('Average Sentiment Polarity')

        # 4. Sentiment components (neg, neu, pos)
        ax4 = plt.subplot(3, 3, 4)
        if all(col in self.all_news.columns for col in ['sentiment_neg', 'sentiment_neu', 'sentiment_pos']):
            sentiment_components = self.all_news[['sentiment_neg', 'sentiment_neu', 'sentiment_pos']].mean()
            sentiment_components.plot(kind='bar', ax=ax4, color=['red', 'gray', 'green'], alpha=0.7)
            ax4.set_title('Average Sentiment Components', fontweight='bold')
            ax4.set_xlabel('Component')
            ax4.set_ylabel('Average Score')
            ax4.tick_params(axis='x', rotation=0)

        # 5. Sentiment over time
        ax5 = plt.subplot(3, 3, (5, 6))
        monthly_sentiment = self.all_news.groupby(self.all_news['date'].dt.to_period('M'))['sentiment_polarity'].mean()
        monthly_sentiment.plot(ax=ax5, linewidth=2, marker='o', markersize=4, color='purple')
        ax5.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax5.set_title('Average Sentiment Over Time (Monthly)', fontweight='bold')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Average Sentiment Polarity')
        ax5.grid(alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Sentiment vs volume correlation
        ax6 = plt.subplot(3, 3, 7)
        daily_sentiment = self.all_news.groupby('date_only')['sentiment_polarity'].mean()
        daily_volume = self.all_news.groupby('date_only').size()
        ax6.scatter(daily_volume, daily_sentiment, alpha=0.3, s=20)
        ax6.set_title('Daily Sentiment vs Volume', fontweight='bold')
        ax6.set_xlabel('Number of Articles')
        ax6.set_ylabel('Average Sentiment')
        ax6.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax6.grid(alpha=0.3)

        # 7. Top stocks by sentiment
        ax7 = plt.subplot(3, 3, 8)
        stock_sentiment = self.all_news.groupby('ticker')['sentiment_polarity'].mean().sort_values().head(10)
        stock_sentiment.plot(kind='barh', ax=ax7, color='coral', alpha=0.7)
        ax7.axvline(0, color='black', linestyle='-', linewidth=1)
        ax7.set_title('Bottom 10 Stocks by Avg Sentiment', fontweight='bold')
        ax7.set_xlabel('Average Sentiment Polarity')

        # 8. Sentiment variability by stock
        ax8 = plt.subplot(3, 3, 9)
        stock_sentiment_std = self.all_news.groupby('ticker')['sentiment_polarity'].std().sort_values(ascending=False).head(15)
        stock_sentiment_std.plot(kind='barh', ax=ax8, color='orange', alpha=0.7)
        ax8.set_title('Top 15 Stocks by Sentiment Variability', fontweight='bold')
        ax8.set_xlabel('Sentiment Std Dev')
        ax8.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / '05_sentiment_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 05_sentiment_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 6: SOURCE & PUBLISHER ANALYSIS
    # ====================================================================

    def analyze_sources(self):
        """Analyze news sources and publishers"""
        print(f"\n{'='*80}")
        print("MODULE 6: SOURCE & PUBLISHER ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Extract domain from link
        def extract_domain(url):
            if pd.isna(url):
                return 'Unknown'
            try:
                parsed = urlparse(str(url))
                domain = parsed.netloc
                # Remove www.
                domain = domain.replace('www.', '')
                return domain
            except:
                return 'Unknown'

        self.all_news['source_domain'] = self.all_news['link'].apply(extract_domain)

        # Source diversity
        unique_sources = self.all_news['source_domain'].nunique()
        results['unique_sources'] = unique_sources

        print(f"\n📰 SOURCE DIVERSITY:")
        print(f"  Total unique sources: {unique_sources}")

        # Top sources
        source_counts = self.all_news['source_domain'].value_counts()
        results['top_sources'] = source_counts.head(20).to_dict()

        print(f"\n📊 TOP 20 NEWS SOURCES:")
        for source, count in source_counts.head(20).items():
            pct = (count / len(self.all_news)) * 100
            print(f"  {source:35s}: {count:6,} articles ({pct:5.2f}%)")

        # Source concentration
        top_5_pct = (source_counts.head(5).sum() / len(self.all_news)) * 100
        top_10_pct = (source_counts.head(10).sum() / len(self.all_news)) * 100

        print(f"\n📈 SOURCE CONCENTRATION:")
        print(f"  Top 5 sources: {top_5_pct:.2f}% of all articles")
        print(f"  Top 10 sources: {top_10_pct:.2f}% of all articles")

        results['top_5_concentration'] = top_5_pct
        results['top_10_concentration'] = top_10_pct

        # Sources by sector
        print(f"\n🏢 TOP SOURCE BY SECTOR:")
        for sector in self.all_news['sector'].unique():
            sector_news = self.all_news[self.all_news['sector'] == sector]
            top_source = sector_news['source_domain'].value_counts().head(1)
            if len(top_source) > 0:
                source_name = top_source.index[0]
                count = top_source.values[0]
                pct = (count / len(sector_news)) * 100
                print(f"  {sector:30s}: {source_name:25s} ({count} articles, {pct:.1f}%)")

        # Publication time patterns
        if 'hour' in self.all_news.columns:
            hourly_dist = self.all_news['hour'].value_counts().sort_index()
            print(f"\n🕐 PUBLICATION TIME PATTERNS:")
            print(f"  Peak hour: {hourly_dist.idxmax()}:00 ({hourly_dist.max()} articles)")
            print(f"  Quietest hour: {hourly_dist.idxmin()}:00 ({hourly_dist.min()} articles)")

            # Market hours vs non-market hours
            market_hours = self.all_news[self.all_news['hour'].between(9, 16)].shape[0]
            market_hours_pct = (market_hours / len(self.all_news)) * 100
            print(f"  Market hours (9am-4pm): {market_hours:,} articles ({market_hours_pct:.2f}%)")

        # Create visualizations
        self._plot_source_analysis(source_counts, hourly_dist if 'hour' in self.all_news.columns else None)

        self.results['sources'] = results
        return results

    def _plot_source_analysis(self, source_counts, hourly_dist):
        """Create source analysis visualizations"""

        fig = plt.figure(figsize=(18, 10))

        # 1. Top 20 sources
        ax1 = plt.subplot(2, 3, 1)
        top_20 = source_counts.head(20)
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_20)))
        top_20.plot(kind='barh', ax=ax1, color=colors)
        ax1.set_title('Top 20 News Sources', fontweight='bold')
        ax1.set_xlabel('Number of Articles')
        ax1.invert_yaxis()

        # 2. Source concentration
        ax2 = plt.subplot(2, 3, 2)
        cumulative_pct = (source_counts.cumsum() / source_counts.sum() * 100)
        ax2.plot(range(1, min(51, len(cumulative_pct)+1)), cumulative_pct.head(50), linewidth=2, marker='o', markersize=4)
        ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.axhline(80, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.set_title('Cumulative Source Coverage', fontweight='bold')
        ax2.set_xlabel('Number of Top Sources')
        ax2.set_ylabel('Cumulative % of Articles')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. Articles per source distribution
        ax3 = plt.subplot(2, 3, 3)
        articles_per_source = source_counts.values
        ax3.hist(articles_per_source, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.set_title('Distribution of Articles per Source', fontweight='bold')
        ax3.set_xlabel('Number of Articles')
        ax3.set_ylabel('Number of Sources')
        ax3.set_yscale('log')

        # 4. Hourly publication pattern
        if hourly_dist is not None:
            ax4 = plt.subplot(2, 3, 4)
            hourly_dist.plot(kind='bar', ax=ax4, color='coral', alpha=0.7)
            ax4.set_title('Articles by Hour of Day', fontweight='bold')
            ax4.set_xlabel('Hour')
            ax4.set_ylabel('Number of Articles')
            ax4.tick_params(axis='x', rotation=0)
            # Highlight market hours
            ax4.axvspan(9, 16, alpha=0.2, color='green', label='Market Hours')
            ax4.legend()

        # 5. Day of week pattern
        ax5 = plt.subplot(2, 3, 5)
        if 'day_of_week' in self.all_news.columns:
            dow_counts = self.all_news['day_of_week'].value_counts().sort_index()
            dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax5.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], color='purple', alpha=0.7)
            ax5.set_xticks(range(7))
            ax5.set_xticklabels(dow_labels)
            ax5.set_title('Articles by Day of Week', fontweight='bold')
            ax5.set_xlabel('Day')
            ax5.set_ylabel('Number of Articles')

        # 6. Source diversity by sector
        ax6 = plt.subplot(2, 3, 6)
        sector_source_diversity = self.all_news.groupby('sector')['source_domain'].nunique().sort_values(ascending=False)
        sector_source_diversity.plot(kind='barh', ax=ax6, color='green', alpha=0.7)
        ax6.set_title('Source Diversity by Sector', fontweight='bold')
        ax6.set_xlabel('Number of Unique Sources')
        ax6.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / '06_source_publisher_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 06_source_publisher_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 7: FALSE POSITIVE/NEGATIVE DETECTION
    # ====================================================================

    def analyze_false_positives_negatives(self):
        """Detect potential false positives and false negatives"""
        print(f"\n{'='*80}")
        print("MODULE 7: FALSE POSITIVE/NEGATIVE DETECTION")
        print(f"{'='*80}")

        results = {}

        # FALSE POSITIVES: News that may not be relevant

        # 1. Ticker only in symbols, not in title
        ticker_in_title = self.all_news.apply(
            lambda row: row['ticker'] in str(row['title']).upper(), axis=1
        )
        ticker_not_in_title = (~ticker_in_title).sum()
        ticker_not_in_title_pct = (ticker_not_in_title / len(self.all_news)) * 100

        results['ticker_not_in_title'] = ticker_not_in_title
        results['ticker_not_in_title_pct'] = ticker_not_in_title_pct

        print(f"\n🚩 POTENTIAL FALSE POSITIVES:")
        print(f"  Ticker NOT in title: {ticker_not_in_title:,} articles ({ticker_not_in_title_pct:.2f}%)")

        # 2. Very low sentiment + generic title
        generic_keywords = ['market', 'stocks', 'dow', 'nasdaq', 's&p', 'index', 'wall street', 'sector']
        generic_pattern = '|'.join(generic_keywords)

        if 'sentiment_polarity' in self.all_news.columns:
            neutral_generic = self.all_news[
                (self.all_news['sentiment_polarity'].abs() < 0.1) &
                (self.all_news['title'].str.contains(generic_pattern, case=False, na=False))
            ]
            neutral_generic_count = len(neutral_generic)
            neutral_generic_pct = (neutral_generic_count / len(self.all_news)) * 100

            results['neutral_generic'] = neutral_generic_count
            results['neutral_generic_pct'] = neutral_generic_pct

            print(f"  Neutral + generic title: {neutral_generic_count:,} articles ({neutral_generic_pct:.2f}%)")

        # 3. Multiple tickers mentioned (primary vs secondary)
        secondary_mention = (self.all_news['num_symbols'] > 2).sum()
        secondary_mention_pct = (secondary_mention / len(self.all_news)) * 100

        results['many_tickers'] = secondary_mention
        results['many_tickers_pct'] = secondary_mention_pct

        print(f"  Many tickers (>2): {secondary_mention:,} articles ({secondary_mention_pct:.2f}%)")

        # 4. Very short content (likely summaries only)
        very_short = (self.all_news['content_length'] < 200).sum()
        very_short_pct = (very_short / len(self.all_news)) * 100

        results['very_short_content'] = very_short
        results['very_short_pct'] = very_short_pct

        print(f"  Very short content (<200 chars): {very_short:,} articles ({very_short_pct:.2f}%)")

        # Combined false positive score
        self.all_news['fp_score'] = (
            (~ticker_in_title).astype(int) +
            (self.all_news['num_symbols'] > 2).astype(int) +
            (self.all_news['content_length'] < 200).astype(int)
        )

        high_fp_risk = (self.all_news['fp_score'] >= 2).sum()
        high_fp_risk_pct = (high_fp_risk / len(self.all_news)) * 100

        results['high_fp_risk'] = high_fp_risk
        results['high_fp_risk_pct'] = high_fp_risk_pct

        print(f"\n⚠️  HIGH FALSE POSITIVE RISK:")
        print(f"  Articles with FP score ≥2: {high_fp_risk:,} ({high_fp_risk_pct:.2f}%)")

        # Show examples
        print(f"\n  Examples of potential false positives:")
        fp_examples = self.all_news[self.all_news['fp_score'] >= 2].head(10)
        for idx, row in fp_examples.iterrows():
            print(f"    [{row['ticker']}] {row['title'][:70]}... (FP score: {row['fp_score']})")

        # FALSE NEGATIVES: Important events we might be missing

        print(f"\n🔍 POTENTIAL FALSE NEGATIVES:")

        # Look for stocks with unusually low coverage
        articles_per_stock = self.all_news.groupby('ticker').size()
        low_coverage = articles_per_stock[articles_per_stock < articles_per_stock.quantile(0.25)]

        print(f"  Stocks with low coverage (<Q1={articles_per_stock.quantile(0.25):.0f} articles):")
        for ticker, count in low_coverage.items():
            name = self.stocks[ticker]['name']
            print(f"    {ticker} ({name}): {count} articles")

        # Days with no news for major stocks
        major_stocks = articles_per_stock.nlargest(10).index
        for ticker in major_stocks:
            stock_news = self.all_news[self.all_news['ticker'] == ticker]
            date_range = pd.date_range(stock_news['date'].min(), stock_news['date'].max(), freq='D')
            news_dates = set(stock_news['date_only'])
            missing_days = len(date_range) - len(news_dates)
            coverage_pct = (len(news_dates) / len(date_range)) * 100

            if coverage_pct < 50:
                print(f"    {ticker}: Only {coverage_pct:.1f}% date coverage ({missing_days} days missing)")

        # Create visualizations
        self._plot_fp_fn_analysis()

        self.results['false_positives_negatives'] = results
        return results

    def _plot_fp_fn_analysis(self):
        """Create false positive/negative analysis visualizations"""

        fig = plt.figure(figsize=(16, 10))

        # 1. False positive score distribution
        ax1 = plt.subplot(2, 3, 1)
        fp_dist = self.all_news['fp_score'].value_counts().sort_index()
        colors = ['green', 'yellow', 'orange', 'red'][:len(fp_dist)]
        fp_dist.plot(kind='bar', ax=ax1, color=colors, alpha=0.7)
        ax1.set_title('False Positive Risk Score Distribution', fontweight='bold')
        ax1.set_xlabel('FP Risk Score')
        ax1.set_ylabel('Number of Articles')
        ax1.tick_params(axis='x', rotation=0)

        # 2. Ticker in title vs not
        ax2 = plt.subplot(2, 3, 2)
        ticker_in_title = self.all_news.apply(
            lambda row: 'In Title' if row['ticker'] in str(row['title']).upper() else 'Not in Title', axis=1
        ).value_counts()
        ticker_in_title.plot(kind='bar', ax=ax2, color=['green', 'red'], alpha=0.7)
        ax2.set_title('Ticker Presence in Title', fontweight='bold')
        ax2.set_xlabel('Presence')
        ax2.set_ylabel('Number of Articles')
        ax2.tick_params(axis='x', rotation=0)

        # 3. Content length vs quality
        ax3 = plt.subplot(2, 3, 3)
        content_bins = [0, 100, 200, 500, 1000, 5000, float('inf')]
        content_labels = ['<100', '100-200', '200-500', '500-1K', '1K-5K', '>5K']
        self.all_news['content_bin'] = pd.cut(self.all_news['content_length'], bins=content_bins, labels=content_labels)
        content_dist = self.all_news['content_bin'].value_counts().sort_index()
        colors_grad = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(content_dist)))
        content_dist.plot(kind='bar', ax=ax3, color=colors_grad, alpha=0.7)
        ax3.set_title('Content Length Distribution', fontweight='bold')
        ax3.set_xlabel('Content Length (chars)')
        ax3.set_ylabel('Number of Articles')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Coverage per stock
        ax4 = plt.subplot(2, 3, 4)
        articles_per_stock = self.all_news.groupby('ticker').size().sort_values()
        q1, q3 = articles_per_stock.quantile([0.25, 0.75])
        colors_stock = ['red' if x < q1 else 'orange' if x < q3 else 'green' for x in articles_per_stock]
        articles_per_stock.plot(kind='barh', ax=ax4, color=colors_stock, alpha=0.6)
        ax4.axvline(q1, color='red', linestyle='--', linewidth=1, alpha=0.5, label=f'Q1: {q1:.0f}')
        ax4.axvline(q3, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Q3: {q3:.0f}')
        ax4.set_title('Article Count per Stock (with quartiles)', fontweight='bold')
        ax4.set_xlabel('Number of Articles')
        ax4.legend()

        # 5. Multi-ticker article distribution
        ax5 = plt.subplot(2, 3, 5)
        if 'num_symbols' in self.all_news.columns:
            symbol_counts = self.all_news['num_symbols'].value_counts().sort_index()
            colors_sym = ['green' if x == 1 else 'yellow' if x == 2 else 'red' for x in symbol_counts.index]
            symbol_counts.plot(kind='bar', ax=ax5, color=colors_sym, alpha=0.7)
            ax5.set_title('Number of Tickers per Article', fontweight='bold')
            ax5.set_xlabel('Number of Tickers')
            ax5.set_ylabel('Number of Articles')
            ax5.tick_params(axis='x', rotation=0)

        # 6. FP risk by sector
        ax6 = plt.subplot(2, 3, 6)
        sector_fp = self.all_news[self.all_news['fp_score'] >= 2].groupby('sector').size()
        sector_total = self.all_news.groupby('sector').size()
        sector_fp_pct = (sector_fp / sector_total * 100).sort_values(ascending=False)
        sector_fp_pct.plot(kind='barh', ax=ax6, color='red', alpha=0.7)
        ax6.set_title('High FP Risk by Sector (%)', fontweight='bold')
        ax6.set_xlabel('Percentage of Articles')
        ax6.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / '07_false_positives_negatives_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 07_false_positives_negatives_analysis.png")
        plt.close()

    # ====================================================================
    # MODULE 8: TEMPORAL PATTERNS
    # ====================================================================

    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in news"""
        print(f"\n{'='*80}")
        print("MODULE 8: TEMPORAL PATTERN ANALYSIS")
        print(f"{'='*80}")

        results = {}

        # Yearly trends
        yearly = self.all_news.groupby('year').size()
        results['yearly_trend'] = yearly.to_dict()

        print(f"\n📅 YEARLY TRENDS:")
        for year, count in yearly.items():
            print(f"  {year}: {count:,} articles")

        # Monthly patterns
        monthly_avg = self.all_news.groupby('month').size()
        results['monthly_pattern'] = monthly_avg.to_dict()

        print(f"\n📆 MONTHLY PATTERNS (Average):")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, count in monthly_avg.items():
            if month <= 12:
                print(f"  {month_names[month-1]}: {count:,} articles")

        # Day of week patterns
        if 'day_of_week' in self.all_news.columns:
            dow = self.all_news['day_of_week'].value_counts().sort_index()
            dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            results['day_of_week_pattern'] = {dow_names[i]: int(dow.get(i, 0)) for i in range(7)}

            print(f"\n📊 DAY OF WEEK PATTERNS:")
            for i, name in enumerate(dow_names):
                count = dow.get(i, 0)
                pct = (count / len(self.all_news)) * 100
                print(f"  {name:10s}: {count:6,} articles ({pct:5.2f}%)")

        # Hour of day patterns
        if 'hour' in self.all_news.columns:
            hourly = self.all_news['hour'].value_counts().sort_index()
            results['hourly_pattern'] = hourly.to_dict()

            print(f"\n🕐 HOURLY PATTERNS (Top hours):")
            for hour, count in hourly.nlargest(10).items():
                print(f"  {hour:02d}:00 - {count:,} articles")

            # Market hours analysis
            market_hours = self.all_news[self.all_news['hour'].between(9, 16)]
            after_hours = self.all_news[~self.all_news['hour'].between(9, 16)]

            market_pct = (len(market_hours) / len(self.all_news)) * 100
            after_pct = (len(after_hours) / len(self.all_news)) * 100

            print(f"\n🕰️  MARKET HOURS vs AFTER HOURS:")
            print(f"  Market hours (9am-4pm): {len(market_hours):,} ({market_pct:.2f}%)")
            print(f"  After hours: {len(after_hours):,} ({after_pct:.2f}%)")

            # Weekend vs weekday
            if 'day_of_week' in self.all_news.columns:
                weekend = self.all_news[self.all_news['day_of_week'].isin([5, 6])]
                weekday = self.all_news[~self.all_news['day_of_week'].isin([5, 6])]

                weekend_pct = (len(weekend) / len(self.all_news)) * 100
                weekday_pct = (len(weekday) / len(self.all_news)) * 100

                print(f"\n📅 WEEKEND vs WEEKDAY:")
                print(f"  Weekday: {len(weekday):,} ({weekday_pct:.2f}%)")
                print(f"  Weekend: {len(weekend):,} ({weekend_pct:.2f}%)")

        # Quarterly earnings patterns
        print(f"\n📊 QUARTERLY PATTERNS (Potential Earnings Seasons):")
        # Typically: Jan, Apr, Jul, Oct
        earnings_months = [1, 4, 7, 10]
        for month in earnings_months:
            month_articles = self.all_news[self.all_news['month'] == month]
            earnings_articles = month_articles[month_articles['cat_Earnings'] == 1] if 'cat_Earnings' in month_articles.columns else pd.DataFrame()
            if len(earnings_articles) > 0:
                pct = (len(earnings_articles) / len(month_articles)) * 100
                print(f"  {month_names[month-1]}: {len(earnings_articles):,} earnings articles ({pct:.2f}% of month)")

        # Create visualizations
        self._plot_temporal_patterns()

        self.results['temporal_patterns'] = results
        return results

    def _plot_temporal_patterns(self):
        """Create temporal pattern visualizations"""

        fig = plt.figure(figsize=(20, 12))

        # 1. Yearly trend
        ax1 = plt.subplot(3, 3, 1)
        yearly = self.all_news.groupby('year').size()
        yearly.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
        ax1.set_title('Articles by Year', fontweight='bold')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Articles')
        ax1.tick_params(axis='x', rotation=0)

        # 2. Monthly pattern
        ax2 = plt.subplot(3, 3, 2)
        monthly = self.all_news.groupby('month').size()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax2.bar(range(1, 13), [monthly.get(i, 0) for i in range(1, 13)], color='coral', alpha=0.7)
        ax2.set_xticks(range(1, 13))
        ax2.set_xticklabels(month_names, rotation=45)
        ax2.set_title('Articles by Month (All Years)', fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Number of Articles')

        # 3. Day of week pattern
        ax3 = plt.subplot(3, 3, 3)
        if 'day_of_week' in self.all_news.columns:
            dow = self.all_news['day_of_week'].value_counts().sort_index()
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            colors_dow = ['blue']*5 + ['red']*2  # Weekdays blue, weekend red
            ax3.bar(range(7), [dow.get(i, 0) for i in range(7)], color=colors_dow, alpha=0.7)
            ax3.set_xticks(range(7))
            ax3.set_xticklabels(dow_names)
            ax3.set_title('Articles by Day of Week', fontweight='bold')
            ax3.set_xlabel('Day')
            ax3.set_ylabel('Number of Articles')

        # 4. Hourly pattern
        ax4 = plt.subplot(3, 3, 4)
        if 'hour' in self.all_news.columns:
            hourly = self.all_news['hour'].value_counts().sort_index()
            hours = range(24)
            counts = [hourly.get(h, 0) for h in hours]
            colors_hour = ['lightblue' if 9 <= h <= 16 else 'gray' for h in hours]
            ax4.bar(hours, counts, color=colors_hour, alpha=0.7)
            ax4.set_title('Articles by Hour (Market hours highlighted)', fontweight='bold')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Number of Articles')
            ax4.axvspan(9, 16, alpha=0.1, color='green')

        # 5. Daily volume over time with trend
        ax5 = plt.subplot(3, 3, (5, 6))
        daily_counts = self.all_news.groupby('date_only').size()
        daily_counts.plot(ax=ax5, linewidth=0.5, alpha=0.5, color='blue')

        # Add moving average
        ma_30 = daily_counts.rolling(window=30, center=True).mean()
        ma_30.plot(ax=ax5, linewidth=2, color='red', label='30-day MA')

        ax5.set_title('Daily News Volume with 30-Day Moving Average', fontweight='bold')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Number of Articles')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. Quarterly pattern
        ax6 = plt.subplot(3, 3, 7)
        quarterly = self.all_news.groupby(self.all_news['date'].dt.quarter).size()
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        ax6.bar(range(1, 5), [quarterly.get(i, 0) for i in range(1, 5)], color='purple', alpha=0.7)
        ax6.set_xticks(range(1, 5))
        ax6.set_xticklabels(quarters)
        ax6.set_title('Articles by Quarter (All Years)', fontweight='bold')
        ax6.set_xlabel('Quarter')
        ax6.set_ylabel('Number of Articles')

        # 7. Heatmap: Month vs Year
        ax7 = plt.subplot(3, 3, 8)
        pivot = self.all_news.groupby(['year', 'month']).size().unstack(fill_value=0)
        sns.heatmap(pivot, annot=False, fmt='d', cmap='YlOrRd', ax=ax7, cbar_kws={'label': 'Articles'})
        ax7.set_title('News Volume Heatmap: Year x Month', fontweight='bold')
        ax7.set_xlabel('Month')
        ax7.set_ylabel('Year')

        # 8. Weekday vs Weekend comparison
        ax8 = plt.subplot(3, 3, 9)
        if 'day_of_week' in self.all_news.columns:
            weekend = self.all_news[self.all_news['day_of_week'].isin([5, 6])]
            weekday = self.all_news[~self.all_news['day_of_week'].isin([5, 6])]
            data = [len(weekday), len(weekend)]
            labels = ['Weekday', 'Weekend']
            colors_wd = ['steelblue', 'coral']
            ax8.pie(data, labels=labels, autopct='%1.1f%%', colors=colors_wd, startangle=90)
            ax8.set_title('Weekday vs Weekend Distribution', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / '08_temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved: 08_temporal_patterns_analysis.png")
        plt.close()

    # ====================================================================
    # MAIN EXECUTION
    # ====================================================================

    def run_full_analysis(self):
        """Run all analysis modules"""
        print("\n" + "="*80)
        print("COMPREHENSIVE NEWS EDA - STARTING FULL ANALYSIS")
        print("="*80)

        # Load data
        self.load_all_news()

        # Run all modules
        self.analyze_volume_frequency()
        self.analyze_content_quality()
        self.analyze_event_categories()
        self.analyze_competitor_crossstock()
        self.analyze_sentiment()
        self.analyze_sources()
        self.analyze_false_positives_negatives()
        self.analyze_temporal_patterns()

        # Save results summary
        self._save_results_summary()

        print(f"\n{'='*80}")
        print("✅ COMPREHENSIVE EDA COMPLETE")
        print(f"{'='*80}")
        print(f"\n📁 All outputs saved to: {self.output_dir.absolute()}")
        print(f"\nGenerated files:")
        for file in sorted(self.output_dir.glob('*.png')):
            print(f"  • {file.name}")
        print(f"  • results_summary.json")

    def _save_results_summary(self):
        """Save results summary to JSON"""
        import json

        summary_file = self.output_dir / 'results_summary.json'

        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = self._make_json_serializable(value)

        with open(summary_file, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✅ Saved: results_summary.json")

    def _make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj


def main():
    """Main execution"""
    eda = ComprehensiveNewsEDA()
    eda.run_full_analysis()


if __name__ == "__main__":
    main()
