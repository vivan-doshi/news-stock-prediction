"""
News Data Explorer
Comprehensive analysis of news data to help filter for event studies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NewsExplorer:
    """Explore and analyze news data for event study filtering"""

    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_news(self, ticker):
        """Load news data for a ticker"""
        # Try different file patterns
        possible_files = [
            f"{ticker}_eodhd_news.csv",
            f"{ticker}_news.csv",
            "news_processed.csv"
        ]

        for filename in possible_files:
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"✓ Loading {filename}")
                df = pd.read_csv(filepath)

                # Standardize date column
                date_cols = [c for c in df.columns if 'date' in c.lower()]
                if date_cols:
                    df['date'] = pd.to_datetime(df[date_cols[0]])

                return df, filename

        raise FileNotFoundError(f"No news file found for {ticker}")

    def analyze_news_structure(self, df, ticker):
        """Analyze the structure and content of news data"""
        print(f"\n{'='*80}")
        print(f"📰 NEWS DATA STRUCTURE FOR {ticker}")
        print(f"{'='*80}")

        # Basic info
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  Total articles: {len(df):,}")
        print(f"  Columns: {', '.join(df.columns)}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Days covered: {(df['date'].max() - df['date'].min()).days}")

        # Check for missing values
        print(f"\n❓ MISSING VALUES:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        for col in df.columns:
            if missing[col] > 0:
                print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")

        # Columns available
        print(f"\n📋 AVAILABLE COLUMNS:")
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample = str(df[col].iloc[0])[:50]
            print(f"  • {col} ({dtype}): {sample}...")

        return df

    def analyze_temporal_distribution(self, df, ticker):
        """Analyze temporal patterns in news"""
        print(f"\n{'='*80}")
        print(f"📅 TEMPORAL DISTRIBUTION FOR {ticker}")
        print(f"{'='*80}")

        # Daily counts
        daily_counts = df.groupby(df['date'].dt.date).size()

        print(f"\n📈 NEWS FREQUENCY:")
        print(f"  Average articles per day: {daily_counts.mean():.1f}")
        print(f"  Median articles per day: {daily_counts.median():.1f}")
        print(f"  Max articles in a day: {daily_counts.max()}")
        print(f"  Min articles in a day: {daily_counts.min()}")
        print(f"  Days with news: {len(daily_counts)}")

        # Top news days
        print(f"\n🔥 TOP 10 NEWS DAYS:")
        top_days = daily_counts.nlargest(10)
        for date, count in top_days.items():
            print(f"  {date}: {count} articles")

        # Visualize
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Daily news over time
        ax1 = axes[0]
        daily_counts.plot(ax=ax1, linewidth=0.8, alpha=0.7)
        ax1.set_title(f'{ticker} News Volume Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Number of Articles')
        ax1.grid(alpha=0.3)

        # Distribution histogram
        ax2 = axes[1]
        daily_counts.hist(bins=50, ax=ax2, edgecolor='black', alpha=0.7)
        ax2.axvline(daily_counts.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {daily_counts.mean():.1f}')
        ax2.axvline(daily_counts.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {daily_counts.median():.1f}')
        ax2.set_title('Distribution of Daily News Volume', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Articles per Day')
        ax2.set_ylabel('Frequency (Number of Days)')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_temporal_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved temporal distribution plot to {ticker}_temporal_distribution.png")
        plt.close()

        return daily_counts

    def analyze_content(self, df, ticker):
        """Analyze news content"""
        print(f"\n{'='*80}")
        print(f"📝 CONTENT ANALYSIS FOR {ticker}")
        print(f"{'='*80}")

        # Check for headline/title column
        title_col = None
        for col in ['headline', 'title', 'Title', 'Headline']:
            if col in df.columns:
                title_col = col
                break

        if not title_col:
            print("  ⚠️  No headline/title column found")
            return

        # Length analysis
        df['title_length'] = df[title_col].astype(str).str.len()
        df['title_words'] = df[title_col].astype(str).str.split().str.len()

        print(f"\n📏 HEADLINE STATISTICS:")
        print(f"  Average characters: {df['title_length'].mean():.1f}")
        print(f"  Average words: {df['title_words'].mean():.1f}")
        print(f"  Median words: {df['title_words'].median():.1f}")

        # Check for content/summary
        content_col = None
        for col in ['summary', 'content', 'description', 'Summary', 'Content']:
            if col in df.columns:
                content_col = col
                break

        if content_col:
            df['content_length'] = df[content_col].astype(str).str.len()
            df['content_words'] = df[content_col].astype(str).str.split().str.len()
            print(f"\n📄 CONTENT STATISTICS:")
            print(f"  Average characters: {df['content_length'].mean():.1f}")
            print(f"  Average words: {df['content_words'].mean():.1f}")

        return df

    def analyze_sentiment(self, df, ticker):
        """Analyze sentiment if available"""
        print(f"\n{'='*80}")
        print(f"😊 SENTIMENT ANALYSIS FOR {ticker}")
        print(f"{'='*80}")

        # Check for sentiment columns
        sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower()]

        if not sentiment_cols:
            print("  ⚠️  No sentiment columns found")
            return df

        print(f"\n📊 SENTIMENT COLUMNS FOUND:")
        for col in sentiment_cols:
            print(f"  • {col}")

        # Analyze each sentiment column
        for col in sentiment_cols:
            print(f"\n🎯 {col.upper()}:")

            # Check if numeric or categorical
            if df[col].dtype in ['float64', 'int64']:
                print(f"  Mean: {df[col].mean():.3f}")
                print(f"  Median: {df[col].median():.3f}")
                print(f"  Std: {df[col].std():.3f}")
                print(f"  Min: {df[col].min():.3f}")
                print(f"  Max: {df[col].max():.3f}")

                # Distribution
                print(f"\n  Distribution:")
                print(f"    Very Negative (<-0.5): {(df[col] < -0.5).sum()} ({(df[col] < -0.5).sum()/len(df)*100:.1f}%)")
                print(f"    Negative (-0.5 to -0.1): {((df[col] >= -0.5) & (df[col] < -0.1)).sum()} ({((df[col] >= -0.5) & (df[col] < -0.1)).sum()/len(df)*100:.1f}%)")
                print(f"    Neutral (-0.1 to 0.1): {((df[col] >= -0.1) & (df[col] <= 0.1)).sum()} ({((df[col] >= -0.1) & (df[col] <= 0.1)).sum()/len(df)*100:.1f}%)")
                print(f"    Positive (0.1 to 0.5): {((df[col] > 0.1) & (df[col] <= 0.5)).sum()} ({((df[col] > 0.1) & (df[col] <= 0.5)).sum()/len(df)*100:.1f}%)")
                print(f"    Very Positive (>0.5): {(df[col] > 0.5).sum()} ({(df[col] > 0.5).sum()/len(df)*100:.1f}%)")
            else:
                print(f"  Value counts:")
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    print(f"    {val}: {count} ({count/len(df)*100:.1f}%)")

        # Visualize sentiment distribution
        if sentiment_cols:
            fig, axes = plt.subplots(1, len(sentiment_cols), figsize=(6*len(sentiment_cols), 5))
            if len(sentiment_cols) == 1:
                axes = [axes]

            for idx, col in enumerate(sentiment_cols):
                ax = axes[idx]
                if df[col].dtype in ['float64', 'int64']:
                    df[col].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
                    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
                    ax.axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.2f}')
                else:
                    df[col].value_counts().plot(kind='bar', ax=ax, alpha=0.7)

                ax.set_title(f'{col} Distribution', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'{ticker}_sentiment_distribution.png', dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved sentiment distribution plot to {ticker}_sentiment_distribution.png")
            plt.close()

        return df

    def analyze_sources(self, df, ticker):
        """Analyze news sources"""
        print(f"\n{'='*80}")
        print(f"📡 NEWS SOURCES FOR {ticker}")
        print(f"{'='*80}")

        # Check for source column
        source_col = None
        for col in ['source', 'Source', 'publisher']:
            if col in df.columns:
                source_col = col
                break

        if not source_col:
            print("  ⚠️  No source column found")
            return df

        source_counts = df[source_col].value_counts()

        print(f"\n📰 TOP 10 NEWS SOURCES:")
        for source, count in source_counts.head(10).items():
            print(f"  {source}: {count} articles ({count/len(df)*100:.1f}%)")

        print(f"\n📊 SOURCE DIVERSITY:")
        print(f"  Total unique sources: {len(source_counts)}")
        print(f"  Articles from top source: {source_counts.iloc[0]} ({source_counts.iloc[0]/len(df)*100:.1f}%)")
        print(f"  Articles from top 5 sources: {source_counts.head(5).sum()} ({source_counts.head(5).sum()/len(df)*100:.1f}%)")

        return df

    def identify_key_events(self, df, ticker):
        """Identify potential key events using keyword analysis"""
        print(f"\n{'='*80}")
        print(f"🔑 KEY EVENT IDENTIFICATION FOR {ticker}")
        print(f"{'='*80}")

        # Check for title/headline
        title_col = None
        for col in ['headline', 'title', 'Title', 'Headline']:
            if col in df.columns:
                title_col = col
                break

        if not title_col:
            print("  ⚠️  No headline/title column found")
            return

        # Define event categories
        event_categories = {
            'Earnings': ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4', 'revenue', 'profit'],
            'Product Launches': ['launch', 'release', 'unveil', 'introduce', 'new product', 'new model', 'announcement'],
            'Executive News': ['ceo', 'executive', 'leadership', 'management', 'board', 'resign', 'appoint', 'hire'],
            'Partnerships': ['partnership', 'deal', 'agreement', 'collaboration', 'acquire', 'acquisition', 'merger'],
            'Regulatory': ['sec', 'lawsuit', 'investigation', 'probe', 'regulator', 'compliance', 'legal'],
            'Market Performance': ['stock', 'shares', 'market', 'trading', 'price', 'rally', 'drop', 'surge', 'plunge'],
        }

        # Find events in each category
        event_results = {}
        for category, keywords in event_categories.items():
            pattern = '|'.join(keywords)
            matches = df[df[title_col].str.contains(pattern, case=False, na=False)]
            event_results[category] = len(matches)
            print(f"\n📌 {category.upper()}: {len(matches)} articles")

            if len(matches) > 0:
                # Show top 5 examples
                print(f"  Examples:")
                for idx, row in matches.head(5).iterrows():
                    date = row['date'].date() if pd.notna(row['date']) else 'Unknown'
                    title = row[title_col][:80]
                    print(f"    • [{date}] {title}...")

        # Create summary visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = list(event_results.keys())
        counts = list(event_results.values())

        bars = ax.bar(categories, counts, alpha=0.7, edgecolor='black')

        # Color bars by count
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_title(f'{ticker} News by Event Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Event Category')
        ax.set_ylabel('Number of Articles')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (cat, count) in enumerate(zip(categories, counts)):
            ax.text(i, count + max(counts)*0.02, str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_event_categories.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved event categories plot to {ticker}_event_categories.png")
        plt.close()

        return event_results

    def suggest_filters(self, df, ticker, daily_counts):
        """Suggest filtering strategies based on analysis"""
        print(f"\n{'='*80}")
        print(f"💡 FILTERING SUGGESTIONS FOR {ticker}")
        print(f"{'='*80}")

        print(f"\n🎯 RECOMMENDED FILTERING STRATEGIES:\n")

        # Strategy 1: High-volume days
        threshold_90 = daily_counts.quantile(0.90)
        threshold_75 = daily_counts.quantile(0.75)
        high_volume_days = (daily_counts >= threshold_90).sum()

        print(f"1. HIGH-VOLUME NEWS DAYS (Top 10%)")
        print(f"   • Filter days with ≥{threshold_90:.0f} articles")
        print(f"   • Results in {high_volume_days} event days")
        print(f"   • Pros: Focuses on major news events")
        print(f"   • Cons: May miss important single-article events")

        # Strategy 2: Sentiment extremes
        sentiment_cols = [c for c in df.columns if 'sentiment' in c.lower()]
        if sentiment_cols:
            print(f"\n2. EXTREME SENTIMENT EVENTS")
            for col in sentiment_cols:
                if df[col].dtype in ['float64', 'int64']:
                    extreme_pos = (df[col] > 0.5).sum()
                    extreme_neg = (df[col] < -0.5).sum()
                    print(f"   • {col} > 0.5 or < -0.5")
                    print(f"   • Results in {extreme_pos + extreme_neg} articles")
                    print(f"   • Pros: Strong sentiment likely to impact stock")
                    print(f"   • Cons: Misses neutral but important news")

        # Strategy 3: Event categories
        print(f"\n3. SPECIFIC EVENT CATEGORIES")
        print(f"   • Focus on: Earnings, Product Launches, Executive News")
        print(f"   • Use keyword filtering from key events analysis")
        print(f"   • Pros: Theoretically important events")
        print(f"   • Cons: Keyword matching may have false positives/negatives")

        # Strategy 4: Combined approach
        print(f"\n4. COMBINED APPROACH (RECOMMENDED)")
        print(f"   • Include if ANY of:")
        print(f"     - High volume day (top 25%: ≥{threshold_75:.0f} articles)")
        print(f"     - Extreme sentiment (if available)")
        print(f"     - Key event keywords (earnings, product, executive)")
        print(f"   • Pros: Comprehensive, less likely to miss events")
        print(f"   • Cons: May include some noise")

        # Strategy 5: One article per day
        print(f"\n5. ONE ARTICLE PER DAY SAMPLING")
        print(f"   • Select highest sentiment/most relevant article per day")
        print(f"   • Results in {len(daily_counts)} event days")
        print(f"   • Pros: Avoids clustering issue")
        print(f"   • Cons: May miss multiple distinct events in one day")

        print(f"\n{'='*80}")
        print(f"💭 NEXT STEPS:")
        print(f"{'='*80}")
        print(f"\n1. Review the generated plots in {self.output_dir}/")
        print(f"2. Choose a filtering strategy based on your research goals")
        print(f"3. Create filtered dataset using chosen strategy")
        print(f"4. Run event study on filtered data")
        print(f"\nExample code to implement Filter Strategy 4:")
        print(f"""
# Combined filtering approach
threshold = daily_counts.quantile(0.75)
daily_counts_dict = daily_counts.to_dict()
df['daily_count'] = df['date'].dt.date.map(daily_counts_dict)

# Create filter
filtered_df = df[
    (df['daily_count'] >= threshold) |  # High volume
    (df['sentiment_polarity'].abs() > 0.5) |  # Extreme sentiment (if available)
    (df['headline'].str.contains('earnings|launch|ceo', case=False, na=False))  # Key events
]

# Save filtered data
filtered_df.to_csv('{ticker}_filtered_news.csv', index=False)
""")

    def run_complete_exploration(self, ticker):
        """Run complete news exploration for a ticker"""
        print(f"\n{'#'*80}")
        print(f"{'#'*80}")
        print(f"###{' '*74}###")
        print(f"###   NEWS DATA EXPLORATION TOOL - {ticker:^44} ###")
        print(f"###{' '*74}###")
        print(f"{'#'*80}")
        print(f"{'#'*80}")

        try:
            # Load data
            df, filename = self.load_news(ticker)

            # Run analyses
            df = self.analyze_news_structure(df, ticker)
            daily_counts = self.analyze_temporal_distribution(df, ticker)
            df = self.analyze_content(df, ticker)
            df = self.analyze_sentiment(df, ticker)
            df = self.analyze_sources(df, ticker)
            event_results = self.identify_key_events(df, ticker)
            self.suggest_filters(df, ticker, daily_counts)

            print(f"\n{'='*80}")
            print(f"✅ EXPLORATION COMPLETE FOR {ticker}")
            print(f"{'='*80}")
            print(f"\n📁 Output files saved to: {self.output_dir.absolute()}/")
            print(f"  • {ticker}_temporal_distribution.png")
            print(f"  • {ticker}_sentiment_distribution.png (if sentiment available)")
            print(f"  • {ticker}_event_categories.png")

            return df, daily_counts, event_results

        except Exception as e:
            print(f"\n❌ Error exploring {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


def main():
    """Run news exploration for both tickers"""
    explorer = NewsExplorer()

    # Explore both tickers
    for ticker in ['TSLA', 'AAPL']:
        df, daily_counts, events = explorer.run_complete_exploration(ticker)
        if df is not None:
            print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
