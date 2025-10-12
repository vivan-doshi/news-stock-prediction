"""
Run Event Study Analysis on Filtered News Data
Uses the filtered news with event categories
"""

import sys
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
main_analysis_module = importlib.import_module('05_main_analysis')
Phase1Analysis = main_analysis_module.Phase1Analysis


class FilteredEventStudy:
    """Event study using filtered news with categories"""

    def __init__(self, data_dir="../01-data", filtered_dir="../03-output/filtered_analysis",
                 output_dir="../03-output/filtered_analysis"):
        self.data_dir = data_dir
        self.filtered_dir = Path(filtered_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def run_event_study(self, ticker):
        """Run event study with filtered event dates"""
        print(f"\n{'='*80}")
        print(f"RUNNING EVENT STUDY FOR {ticker} (FILTERED DATA)")
        print(f"{'='*80}")

        # File paths
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = self.filtered_dir / f"{ticker}_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = str(self.output_dir / f"{ticker}_event_study")

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        print(f"\n📂 Input Files:")
        print(f"  Stock data: {stock_file}")
        print(f"  Event dates: {event_dates_file.name}")
        print(f"  Fama-French: {ff_file}")
        print(f"  Output dir: {output_dir}")

        try:
            # Run analysis
            analysis = Phase1Analysis(
                stock_file=stock_file,
                news_file=str(event_dates_file),
                ff_file=ff_file,
                sector_file=None,
                data_dir=self.data_dir,
                output_dir=output_dir
            )

            summary = analysis.run_complete_analysis()

            if summary:
                print(f"\n✅ {ticker} Event Study Complete!")
                print(f"\n📊 Key Results:")
                print(f"  Event Days: {summary['news_days']}")
                print(f"  Non-Event Days: {summary['non_news_days']}")
                print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.3f}%")
                print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.3f}%")
                print(f"  Difference: {(summary['mean_ar_news'] - summary['mean_ar_non_news'])*100:.3f}%")
                print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                print(f"  Model R²: {summary['avg_r_squared']:.3f}")

                return summary
            else:
                print(f"\n❌ {ticker} Event Study Failed!")
                return None

        except Exception as e:
            print(f"\n❌ {ticker} Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_by_category(self, ticker, summary):
        """Analyze event impact by news category"""
        print(f"\n{'='*80}")
        print(f"CATEGORY-SPECIFIC ANALYSIS FOR {ticker}")
        print(f"{'='*80}")

        try:
            # Load filtered news with categories
            filtered_news = pd.read_csv(self.filtered_dir / f"{ticker}_news_filtered.csv")
            filtered_news['date'] = pd.to_datetime(filtered_news['date']).dt.date

            # Load abnormal returns
            ar_file = Path(self.output_dir) / f"{ticker}_event_study" / "abnormal_returns.csv"
            if not ar_file.exists():
                print("  ⚠️  Abnormal returns file not found")
                return None

            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
            ar_df['date'] = ar_df.index.date

            # Merge with news categories
            # For each date, get the primary category
            date_categories = filtered_news.groupby('date')['primary_category'].agg(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 'general'
            )

            ar_df['primary_category'] = ar_df['date'].map(date_categories)
            ar_df['primary_category'] = ar_df['primary_category'].fillna('no_news')

            # Analyze by category
            print(f"\n📋 ABNORMAL RETURNS BY CATEGORY:")
            print(f"\n{'Category':<15} {'Count':>8} {'Mean AR':>10} {'Std AR':>10} {'t-stat':>8} {'p-value':>8}")
            print("-" * 80)

            category_results = {}

            for category in filtered_news['primary_category'].unique():
                cat_data = ar_df[ar_df['primary_category'] == category]['Abnormal_Return'].dropna()
                no_news_data = ar_df[ar_df['primary_category'] == 'no_news']['Abnormal_Return'].dropna()

                if len(cat_data) > 5 and len(no_news_data) > 5:
                    mean_ar = cat_data.mean()
                    std_ar = cat_data.std()

                    # T-test vs non-news days
                    t_stat, p_val = stats.ttest_ind(cat_data, no_news_data, equal_var=False)

                    category_results[category] = {
                        'count': len(cat_data),
                        'mean_ar': mean_ar,
                        'std_ar': std_ar,
                        't_stat': t_stat,
                        'p_value': p_val
                    }

                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
                    print(f"{category:<15} {len(cat_data):>8,} {mean_ar*100:>9.3f}% {std_ar*100:>9.3f}% {t_stat:>8.2f} {p_val:>8.4f} {sig}")

            # Visualize category-specific results
            self.visualize_category_analysis(ticker, category_results, ar_df)

            return category_results

        except Exception as e:
            print(f"  ❌ Error analyzing by category: {e}")
            import traceback
            traceback.print_exc()
            return None

    def visualize_category_analysis(self, ticker, category_results, ar_df):
        """Create visualizations for category-specific analysis"""
        print(f"\n📊 Creating category analysis visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean abnormal returns by category
        ax1 = axes[0, 0]
        categories = list(category_results.keys())
        mean_ars = [category_results[cat]['mean_ar'] * 100 for cat in categories]
        colors = ['green' if ar > 0 else 'red' for ar in mean_ars]

        bars = ax1.barh(categories, mean_ars, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linewidth=1)
        ax1.set_title('Mean Abnormal Returns by Event Category', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Abnormal Return (%)')
        ax1.grid(axis='x', alpha=0.3)

        # Add significance markers
        for i, cat in enumerate(categories):
            p_val = category_results[cat]['p_value']
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            if sig:
                x_pos = mean_ars[i] + (0.01 if mean_ars[i] > 0 else -0.01)
                ax1.text(x_pos, i, sig, va='center', fontweight='bold', fontsize=14)

        # 2. T-statistics by category
        ax2 = axes[0, 1]
        t_stats = [category_results[cat]['t_stat'] for cat in categories]
        colors = ['green' if t > 0 else 'red' for t in t_stats]

        ax2.barh(categories, t_stats, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='black', linewidth=1)
        ax2.axvline(1.96, color='blue', linestyle='--', linewidth=2, label='5% significance')
        ax2.axvline(-1.96, color='blue', linestyle='--', linewidth=2)
        ax2.set_title('T-Statistics by Event Category', fontsize=12, fontweight='bold')
        ax2.set_xlabel('T-Statistic')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        # 3. Distribution of ARs by category
        ax3 = axes[1, 0]
        for category in categories:
            cat_data = ar_df[ar_df['primary_category'] == category]['Abnormal_Return'].dropna()
            ax3.hist(cat_data * 100, bins=50, alpha=0.5, label=category, density=True)

        ax3.axvline(0, color='black', linewidth=2, linestyle='--')
        ax3.set_title('Distribution of Abnormal Returns by Category', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Abnormal Return (%)')
        ax3.set_ylabel('Density')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        # 4. Sample size and significance
        ax4 = axes[1, 1]
        sample_sizes = [category_results[cat]['count'] for cat in categories]
        p_values = [category_results[cat]['p_value'] for cat in categories]

        scatter = ax4.scatter(sample_sizes, mean_ars, s=[1000/p if p > 0.001 else 1000 for p in p_values],
                            c=colors, alpha=0.6, edgecolors='black', linewidth=2)

        for i, cat in enumerate(categories):
            ax4.annotate(cat, (sample_sizes[i], mean_ars[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax4.axhline(0, color='black', linewidth=1, linestyle='--')
        ax4.set_title('Effect Size vs Sample Size\n(bubble size = significance)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Number of Events')
        ax4.set_ylabel('Mean Abnormal Return (%)')
        ax4.grid(alpha=0.3)

        plt.suptitle(f'{ticker} Event Study: Category-Specific Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.output_dir / f'{ticker}_category_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_file.name}")
        plt.close()

    def create_comprehensive_summary(self, results):
        """Create comprehensive summary of all results"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVENT STUDY SUMMARY")
        print(f"{'='*80}")

        summary_text = []
        summary_text.append("=" * 80)
        summary_text.append("EVENT STUDY RESULTS - FILTERED NEWS DATA")
        summary_text.append("=" * 80)
        summary_text.append("")

        for ticker, data in results.items():
            if data and data['summary']:
                summary = data['summary']
                summary_text.append(f"\n{ticker} EVENT STUDY:")
                summary_text.append("-" * 80)
                summary_text.append(f"  Event Days: {summary['news_days']:,}")
                summary_text.append(f"  Non-Event Days: {summary['non_news_days']:,}")
                summary_text.append(f"  Mean AR (Events): {summary['mean_ar_news']*100:.3f}%")
                summary_text.append(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.3f}%")
                summary_text.append(f"  Difference: {(summary['mean_ar_news'] - summary['mean_ar_non_news'])*100:.3f}%")
                summary_text.append(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                summary_text.append(f"  Model R²: {summary['avg_r_squared']:.3f}")

                # Interpretation
                if summary['significant_tests'] >= 2:
                    summary_text.append(f"  ✅ RESULT: Strong evidence of event impact")
                elif summary['significant_tests'] >= 1:
                    summary_text.append(f"  ⚠️  RESULT: Moderate evidence of event impact")
                else:
                    summary_text.append(f"  ❌ RESULT: Limited evidence of event impact")

                # Category results
                if data['categories']:
                    summary_text.append(f"\n  CATEGORY-SPECIFIC RESULTS:")
                    for category, cat_data in data['categories'].items():
                        sig = "***" if cat_data['p_value'] < 0.01 else "**" if cat_data['p_value'] < 0.05 else "*" if cat_data['p_value'] < 0.10 else ""
                        summary_text.append(f"    {category:<15}: {cat_data['mean_ar']*100:>7.3f}% (p={cat_data['p_value']:.4f}) {sig}")

                summary_text.append("")

        summary_text.append("=" * 80)
        summary_text.append("NOTES:")
        summary_text.append("  *** p < 0.01, ** p < 0.05, * p < 0.10")
        summary_text.append("=" * 80)

        # Save summary
        summary_file = self.output_dir / "comprehensive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("\n".join(summary_text))

        print("\n".join(summary_text))
        print(f"\n✓ Saved comprehensive summary: {summary_file.name}")

    def run_complete_analysis(self):
        """Run complete event study analysis for both tickers"""
        print("\n" + "#"*80)
        print("#"*80)
        print("###" + " "*74 + "###")
        print("###" + "   FILTERED EVENT STUDY ANALYSIS".center(74) + "###")
        print("###" + " "*74 + "###")
        print("#"*80)
        print("#"*80)

        results = {}

        for ticker in ['TSLA', 'AAPL']:
            print(f"\n\n{'='*80}")
            print(f"PROCESSING {ticker}")
            print(f"{'='*80}")

            # Run event study
            summary = self.run_event_study(ticker)

            # Analyze by category
            categories = None
            if summary:
                categories = self.analyze_by_category(ticker, summary)

            results[ticker] = {
                'summary': summary,
                'categories': categories
            }

        # Create comprehensive summary
        self.create_comprehensive_summary(results)

        print(f"\n{'='*80}")
        print(f"✅ ALL ANALYSES COMPLETE!")
        print(f"{'='*80}")
        print(f"\n📁 Results saved to: {self.output_dir.absolute()}")
        print(f"\nGenerated files:")
        print(f"  • TSLA_event_study/ (standard event study outputs)")
        print(f"  • AAPL_event_study/ (standard event study outputs)")
        print(f"  • TSLA_category_analysis.png")
        print(f"  • AAPL_category_analysis.png")
        print(f"  • comprehensive_summary.txt")
        print(f"{'='*80}")

        return results


def main():
    """Run the complete filtered event study"""
    study = FilteredEventStudy()
    results = study.run_complete_analysis()
    return results


if __name__ == "__main__":
    main()
