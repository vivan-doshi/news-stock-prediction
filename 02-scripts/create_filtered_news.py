"""
Create Filtered News Datasets
Combines event keywords and sentiment filtering with category tagging
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class NewsFilter:
    """Filter news data based on events and sentiment, adding category columns"""

    def __init__(self, data_dir="../01-data", output_dir="../03-output/filtered_analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Define event categories with keywords
        self.event_categories = {
            'earnings': ['earnings', 'quarterly results', 'financial results',
                        'q1', 'q2', 'q3', 'q4', 'revenue', 'profit', 'eps'],
            'product': ['launch', 'release', 'unveil', 'introduce',
                       'new product', 'new model', 'announcement'],
            'executive': ['ceo', 'executive', 'leadership', 'management',
                         'board', 'resign', 'appoint', 'hire', 'chief'],
            'partnership': ['partnership', 'deal', 'agreement', 'collaboration',
                          'acquire', 'acquisition', 'merger', 'joint venture'],
            'regulatory': ['sec', 'lawsuit', 'investigation', 'probe',
                          'regulator', 'compliance', 'legal', 'fine'],
            'market': ['stock', 'shares', 'market', 'trading', 'price',
                      'rally', 'drop', 'surge', 'plunge', 'investors']
        }

    def load_news(self, ticker):
        """Load news data for a ticker"""
        filename = f"{ticker}_eodhd_news.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        print(f"✓ Loading {filename}")
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])

        return df

    def categorize_news(self, df):
        """Add category columns to identify event types"""
        print("\n📋 Categorizing news articles...")

        # Initialize category columns
        for category in self.event_categories.keys():
            df[f'category_{category}'] = False

        # Check title and content for each category
        for category, keywords in self.event_categories.items():
            pattern = '|'.join(keywords)

            # Check both title and content
            title_match = df['title'].str.contains(pattern, case=False, na=False)
            content_match = df['content'].str.contains(pattern, case=False, na=False)

            df[f'category_{category}'] = title_match | content_match

            count = df[f'category_{category}'].sum()
            print(f"  • {category.capitalize()}: {count:,} articles ({count/len(df)*100:.1f}%)")

        # Create a primary category (first matched category)
        def get_primary_category(row):
            for category in self.event_categories.keys():
                if row[f'category_{category}']:
                    return category
            return 'general'

        df['primary_category'] = df.apply(get_primary_category, axis=1)

        # Count total categories per article
        category_cols = [f'category_{cat}' for cat in self.event_categories.keys()]
        df['num_categories'] = df[category_cols].sum(axis=1)

        print(f"\n📊 Category Statistics:")
        print(f"  Articles with 0 categories: {(df['num_categories'] == 0).sum():,}")
        print(f"  Articles with 1 category: {(df['num_categories'] == 1).sum():,}")
        print(f"  Articles with 2+ categories: {(df['num_categories'] >= 2).sum():,}")
        print(f"  Max categories in one article: {df['num_categories'].max()}")

        return df

    def apply_filters(self, df, ticker):
        """Apply combined event and sentiment filtering"""
        print(f"\n🔍 Applying filters for {ticker}...")

        # Calculate daily article counts
        daily_counts = df.groupby(df['date'].dt.date).size()
        threshold_75 = daily_counts.quantile(0.75)

        # Map daily counts to dataframe
        df['daily_count'] = df['date'].dt.date.map(daily_counts.to_dict())

        print(f"\n📈 Filter Thresholds:")
        print(f"  High-volume threshold (75th percentile): {threshold_75:.0f} articles/day")
        print(f"  Extreme sentiment threshold: |polarity| > 0.5")
        print(f"  Event keywords: All categories defined")

        # Create filter criteria
        high_volume = df['daily_count'] >= threshold_75
        extreme_sentiment = df['sentiment_polarity'].abs() > 0.5

        # Event keyword filter - at least one category match
        has_event_keyword = df['num_categories'] > 0

        # Combined filter: ANY of the three conditions
        df['filter_high_volume'] = high_volume
        df['filter_extreme_sentiment'] = extreme_sentiment
        df['filter_event_keyword'] = has_event_keyword
        df['passes_filter'] = high_volume | extreme_sentiment | has_event_keyword

        # Statistics
        print(f"\n📊 Filter Results:")
        print(f"  High-volume days: {high_volume.sum():,} articles ({high_volume.sum()/len(df)*100:.1f}%)")
        print(f"  Extreme sentiment: {extreme_sentiment.sum():,} articles ({extreme_sentiment.sum()/len(df)*100:.1f}%)")
        print(f"  Event keywords: {has_event_keyword.sum():,} articles ({has_event_keyword.sum()/len(df)*100:.1f}%)")
        print(f"  Combined (passes filter): {df['passes_filter'].sum():,} articles ({df['passes_filter'].sum()/len(df)*100:.1f}%)")

        # Filter statistics by category
        filtered_df = df[df['passes_filter']]
        print(f"\n📋 Filtered Data by Primary Category:")
        category_dist = filtered_df['primary_category'].value_counts()
        for category, count in category_dist.items():
            print(f"  • {category.capitalize()}: {count:,} ({count/len(filtered_df)*100:.1f}%)")

        # Date statistics
        print(f"\n📅 Filtered Data Coverage:")
        unique_dates = filtered_df['date'].dt.date.nunique()
        total_dates = df['date'].dt.date.nunique()
        print(f"  Days with news: {unique_dates} out of {total_dates} ({unique_dates/total_dates*100:.1f}%)")
        print(f"  Articles per day (filtered): {len(filtered_df)/unique_dates:.1f} avg")

        return df, filtered_df

    def save_datasets(self, df, filtered_df, ticker):
        """Save both full and filtered datasets"""
        print(f"\n💾 Saving datasets for {ticker}...")

        # Save full dataset with categories and filters
        full_output = self.output_dir / f"{ticker}_news_categorized.csv"
        df.to_csv(full_output, index=False)
        print(f"  ✓ Saved full categorized dataset: {full_output.name} ({len(df):,} articles)")

        # Save filtered dataset
        filtered_output = self.output_dir / f"{ticker}_news_filtered.csv"
        filtered_df.to_csv(filtered_output, index=False)
        print(f"  ✓ Saved filtered dataset: {filtered_output.name} ({len(filtered_df):,} articles)")

        # Create event dates file for event study
        event_dates = pd.DataFrame({'Date': filtered_df['date'].dt.date.unique()})
        event_dates = event_dates.sort_values('Date')
        dates_output = self.output_dir / f"{ticker}_event_dates.csv"
        event_dates.to_csv(dates_output, index=False)
        print(f"  ✓ Saved event dates: {dates_output.name} ({len(event_dates)} unique dates)")

        # Save summary statistics
        summary = self.create_summary(df, filtered_df, ticker)
        summary_output = self.output_dir / f"{ticker}_filter_summary.txt"
        with open(summary_output, 'w') as f:
            f.write(summary)
        print(f"  ✓ Saved summary: {summary_output.name}")

        return full_output, filtered_output, dates_output

    def create_summary(self, df, filtered_df, ticker):
        """Create a text summary of the filtering process"""
        summary = []
        summary.append("=" * 80)
        summary.append(f"NEWS FILTERING SUMMARY FOR {ticker}")
        summary.append("=" * 80)
        summary.append("")

        summary.append("ORIGINAL DATASET:")
        summary.append(f"  Total articles: {len(df):,}")
        summary.append(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        summary.append(f"  Days with news: {df['date'].dt.date.nunique():,}")
        summary.append(f"  Articles per day: {len(df)/df['date'].dt.date.nunique():.1f} avg")
        summary.append("")

        summary.append("FILTERING CRITERIA:")
        summary.append("  1. High-volume days (top 25% by article count)")
        summary.append("  2. Extreme sentiment (|polarity| > 0.5)")
        summary.append("  3. Event keywords (earnings, product, executive, etc.)")
        summary.append("  Filter passes if ANY condition is met")
        summary.append("")

        summary.append("FILTERED DATASET:")
        summary.append(f"  Total articles: {len(filtered_df):,} ({len(filtered_df)/len(df)*100:.1f}% of original)")
        summary.append(f"  Unique dates: {filtered_df['date'].dt.date.nunique():,}")
        summary.append(f"  Articles per day: {len(filtered_df)/filtered_df['date'].dt.date.nunique():.1f} avg")
        summary.append("")

        summary.append("FILTER BREAKDOWN:")
        summary.append(f"  High-volume: {df['filter_high_volume'].sum():,} articles")
        summary.append(f"  Extreme sentiment: {df['filter_extreme_sentiment'].sum():,} articles")
        summary.append(f"  Event keywords: {df['filter_event_keyword'].sum():,} articles")
        summary.append("")

        summary.append("CATEGORY DISTRIBUTION (Filtered Data):")
        category_dist = filtered_df['primary_category'].value_counts()
        for category, count in category_dist.items():
            pct = count/len(filtered_df)*100
            summary.append(f"  {category.capitalize():15s}: {count:6,} ({pct:5.1f}%)")
        summary.append("")

        summary.append("SENTIMENT DISTRIBUTION (Filtered Data):")
        summary.append(f"  Mean polarity: {filtered_df['sentiment_polarity'].mean():.3f}")
        summary.append(f"  Std polarity: {filtered_df['sentiment_polarity'].std():.3f}")
        summary.append(f"  Very negative (<-0.5): {(filtered_df['sentiment_polarity'] < -0.5).sum():,}")
        summary.append(f"  Negative (-0.5 to -0.1): {((filtered_df['sentiment_polarity'] >= -0.5) & (filtered_df['sentiment_polarity'] < -0.1)).sum():,}")
        summary.append(f"  Neutral (-0.1 to 0.1): {((filtered_df['sentiment_polarity'] >= -0.1) & (filtered_df['sentiment_polarity'] <= 0.1)).sum():,}")
        summary.append(f"  Positive (0.1 to 0.5): {((filtered_df['sentiment_polarity'] > 0.1) & (filtered_df['sentiment_polarity'] <= 0.5)).sum():,}")
        summary.append(f"  Very positive (>0.5): {(filtered_df['sentiment_polarity'] > 0.5).sum():,}")
        summary.append("")

        summary.append("=" * 80)
        summary.append("Files created:")
        summary.append(f"  - {ticker}_news_categorized.csv (full dataset with categories)")
        summary.append(f"  - {ticker}_news_filtered.csv (filtered dataset)")
        summary.append(f"  - {ticker}_event_dates.csv (for event study)")
        summary.append("=" * 80)

        return "\n".join(summary)

    def process_ticker(self, ticker):
        """Complete processing for one ticker"""
        print(f"\n{'='*80}")
        print(f"PROCESSING {ticker}")
        print(f"{'='*80}")

        # Load data
        df = self.load_news(ticker)

        # Add categories
        df = self.categorize_news(df)

        # Apply filters
        df, filtered_df = self.apply_filters(df, ticker)

        # Save datasets
        files = self.save_datasets(df, filtered_df, ticker)

        print(f"\n✅ {ticker} processing complete!")

        return df, filtered_df, files

    def create_visualizations(self, ticker):
        """Create visualization comparing filtered vs original data"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(f"\n📊 Creating visualizations for {ticker}...")

        # Load the categorized data
        df = pd.read_csv(self.output_dir / f"{ticker}_news_categorized.csv")
        df['date'] = pd.to_datetime(df['date'])
        filtered_df = df[df['passes_filter']]

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Daily article count comparison
        ax1 = fig.add_subplot(gs[0, :])
        daily_all = df.groupby(df['date'].dt.date).size()
        daily_filtered = filtered_df.groupby(filtered_df['date'].dt.date).size()

        ax1.plot(daily_all.index, daily_all.values, alpha=0.5, linewidth=0.8, label='All News', color='gray')
        ax1.plot(daily_filtered.index, daily_filtered.values, alpha=0.8, linewidth=1.5, label='Filtered News', color='blue')
        ax1.set_title(f'{ticker} News Volume: Original vs Filtered', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Articles per Day')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Category distribution
        ax2 = fig.add_subplot(gs[1, 0])
        category_dist = filtered_df['primary_category'].value_counts()
        colors = plt.cm.Set3(range(len(category_dist)))
        ax2.barh(category_dist.index, category_dist.values, color=colors, edgecolor='black', alpha=0.8)
        ax2.set_title('Filtered News by Category', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Articles')
        for i, (cat, val) in enumerate(category_dist.items()):
            ax2.text(val, i, f' {val:,}', va='center', fontweight='bold')

        # 3. Sentiment distribution comparison
        ax3 = fig.add_subplot(gs[1, 1])
        bins = np.linspace(-1, 1, 30)
        ax3.hist(df['sentiment_polarity'], bins=bins, alpha=0.5, label='All News', color='gray', edgecolor='black')
        ax3.hist(filtered_df['sentiment_polarity'], bins=bins, alpha=0.7, label='Filtered News', color='blue', edgecolor='black')
        ax3.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold (±0.5)')
        ax3.axvline(-0.5, color='green', linestyle='--', linewidth=2)
        ax3.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Sentiment Polarity')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. Filter criteria overlap (Venn-like bar chart)
        ax4 = fig.add_subplot(gs[2, 0])
        filter_stats = {
            'High Volume': df['filter_high_volume'].sum(),
            'Extreme Sentiment': df['filter_extreme_sentiment'].sum(),
            'Event Keywords': df['filter_event_keyword'].sum(),
            'Combined (Unique)': df['passes_filter'].sum()
        }
        bars = ax4.bar(filter_stats.keys(), filter_stats.values(),
                       color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                       edgecolor='black', alpha=0.8)
        ax4.set_title('Filter Criteria Results', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Articles')
        ax4.tick_params(axis='x', rotation=45)
        for i, (k, v) in enumerate(filter_stats.items()):
            ax4.text(i, v, f'\n{v:,}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom', fontweight='bold')

        # 5. Multi-category articles
        ax5 = fig.add_subplot(gs[2, 1])
        num_cats = filtered_df['num_categories'].value_counts().sort_index()
        ax5.bar(num_cats.index, num_cats.values, color='purple', edgecolor='black', alpha=0.7)
        ax5.set_title('Articles by Number of Categories', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Number of Categories Tagged')
        ax5.set_ylabel('Number of Articles')
        ax5.grid(axis='y', alpha=0.3)
        for i, (cat, val) in enumerate(num_cats.items()):
            ax5.text(cat, val, f'{val:,}', ha='center', va='bottom', fontweight='bold')

        plt.suptitle(f'{ticker} News Filtering Analysis', fontsize=16, fontweight='bold', y=0.995)

        # Save figure
        output_file = self.output_dir / f'{ticker}_filtering_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved visualization: {output_file.name}")
        plt.close()


def main():
    """Process both tickers"""
    print("\n" + "="*80)
    print("NEWS FILTERING WITH EVENT CATEGORIES AND SENTIMENT")
    print("="*80)

    filter_tool = NewsFilter()

    results = {}

    # Process both tickers
    for ticker in ['TSLA', 'AAPL']:
        try:
            df, filtered_df, files = filter_tool.process_ticker(ticker)
            filter_tool.create_visualizations(ticker)
            results[ticker] = {
                'original': len(df),
                'filtered': len(filtered_df),
                'dates': filtered_df['date'].dt.date.nunique(),
                'files': files
            }
        except Exception as e:
            print(f"\n❌ Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            results[ticker] = None

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    for ticker, result in results.items():
        if result:
            print(f"\n{ticker}:")
            print(f"  Original articles: {result['original']:,}")
            print(f"  Filtered articles: {result['filtered']:,} ({result['filtered']/result['original']*100:.1f}%)")
            print(f"  Event dates: {result['dates']:,}")
            print(f"  ✅ Ready for event study analysis")

    print("\n" + "="*80)
    print(f"📁 All files saved to: {filter_tool.output_dir.absolute()}")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the filtering analysis visualizations")
    print("2. Check the filter summaries")
    print("3. Run event study using the generated event_dates.csv files")
    print("="*80)


if __name__ == "__main__":
    main()
