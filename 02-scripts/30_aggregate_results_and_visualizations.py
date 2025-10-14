"""
Aggregate Event Study Results and Create Comprehensive Visualizations
====================================================================

This script:
1. Collects results from all 50 stocks
2. Creates comprehensive summary tables
3. Generates cross-stock comparison visualizations
4. Creates sector-level analysis charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "03-output" / "balanced_event_study"
SUMMARY_DIR = OUTPUT_DIR / "aggregate_summary"
SUMMARY_DIR.mkdir(exist_ok=True)

# Sector mapping
SECTOR_MAPPING = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'AVGO': 'Technology', 'ORCL': 'Technology',

    'JPM': 'Finance', 'GS': 'Finance', 'BAC': 'Finance',
    'WFC': 'Finance', 'MS': 'Finance',

    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare',

    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary',
    'HD': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'NKE': 'Consumer Discretionary',

    'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples',
    'WMT': 'Consumer Staples',

    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'NEE': 'Energy',

    'BA': 'Industrials', 'UNP': 'Industrials', 'LMT': 'Industrials',
    'RTX': 'Industrials', 'HON': 'Industrials',

    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'SPG': 'Real Estate',
    'EQIX': 'Real Estate', 'CCI': 'Real Estate',

    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials',
    'ECL': 'Materials', 'DD': 'Materials',

    'GOOGL': 'Communication', 'META': 'Communication', 'DIS': 'Communication',
    'CMCSA': 'Communication', 'VZ': 'Communication',
}

print("="*80)
print("AGGREGATE EVENT STUDY ANALYSIS - ALL 50 STOCKS")
print("="*80)
print()

# Step 1: Collect all results
print("[1/5] Collecting results from all stocks...")
all_results = []

for ticker, sector in SECTOR_MAPPING.items():
    summary_file = OUTPUT_DIR / ticker / "summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['ticker'] = ticker
        df['sector'] = sector
        all_results.append(df)
        print(f"  ✓ {ticker}")
    else:
        print(f"  ✗ {ticker} - No results found")

# Combine all results
results_df = pd.concat(all_results, ignore_index=True)
print(f"\nCollected results from {len(all_results)} stocks")
print()

# Step 2: Create summary tables
print("[2/5] Creating summary tables...")

# Calculate derived columns
results_df['difference'] = results_df['mean_ar_news'] - results_df['mean_ar_non_news']
results_df['news_days_pct'] = (results_df['news_days'] / results_df['total_days'] * 100)
results_df['significant'] = results_df['significant_ttest']

# Overall summary
summary = results_df[['ticker', 'sector', 'mean_ar_news', 'mean_ar_non_news',
                       'difference', 'p_value_ttest', 'significant',
                       'cohens_d', 'news_days_pct']].copy()

summary.columns = ['Ticker', 'Sector', 'AR (News)', 'AR (Non-News)',
                   'Difference', 'P-Value', 'Significant', 'Cohen\'s d',
                   'News Days %']

# Sort by absolute difference
summary = summary.sort_values('Difference', ascending=False)

# Save full summary
summary.to_csv(SUMMARY_DIR / "full_summary.csv", index=False)
print(f"  Saved: full_summary.csv")

# Significant results only
sig_summary = summary[summary['Significant'] == True].copy()
sig_summary.to_csv(SUMMARY_DIR / "significant_results.csv", index=False)
print(f"  Saved: significant_results.csv ({len(sig_summary)} stocks)")

# Sector summary
sector_summary = results_df.groupby('sector').agg({
    'mean_ar_news': 'mean',
    'mean_ar_non_news': 'mean',
    'difference': 'mean',
    'significant': 'sum',
    'ticker': 'count'
}).round(4)
sector_summary.columns = ['Avg AR (News)', 'Avg AR (Non-News)',
                          'Avg Difference', 'Significant Count', 'Total Stocks']
sector_summary = sector_summary.sort_values('Avg Difference', ascending=False)
sector_summary.to_csv(SUMMARY_DIR / "sector_summary.csv")
print(f"  Saved: sector_summary.csv")
print()

# Step 3: Create visualizations
print("[3/5] Creating cross-stock comparison visualizations...")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2ecc71' if x else '#e74c3c' for x in summary['Significant']]

# Figure 1: AR Comparison by Stock
fig, ax = plt.subplots(figsize=(20, 12))

x = np.arange(len(summary))
width = 0.35

bars1 = ax.bar(x - width/2, summary['AR (News)'] * 100, width,
               label='News Days', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, summary['AR (Non-News)'] * 100, width,
               label='Non-News Days', alpha=0.8, color='#95a5a6')

ax.set_xlabel('Stock', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Abnormal Return (%)', fontsize=14, fontweight='bold')
ax.set_title('Abnormal Returns: News vs Non-News Days - All 50 Stocks',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(summary['Ticker'], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "01_ar_comparison_all_stocks.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 01_ar_comparison_all_stocks.png")

# Figure 2: Effect Size (Difference) by Stock
fig, ax = plt.subplots(figsize=(20, 10))

bars = ax.barh(range(len(summary)), summary['Difference'] * 100, color=colors, alpha=0.8)

ax.set_yticks(range(len(summary)))
ax.set_yticklabels(summary['Ticker'], fontsize=10)
ax.set_xlabel('Difference in AR (News - Non-News) (%)', fontsize=14, fontweight='bold')
ax.set_title('News Impact Effect Size - All 50 Stocks',
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', alpha=0.8, label='Significant (p < 0.05)'),
    Patch(facecolor='#e74c3c', alpha=0.8, label='Not Significant')
]
ax.legend(handles=legend_elements, fontsize=12, loc='lower right')

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "02_effect_size_all_stocks.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 02_effect_size_all_stocks.png")

# Figure 3: P-Value Distribution
fig, ax = plt.subplots(figsize=(14, 8))

# Log scale for better visualization
p_values = summary['P-Value'].values
p_values_safe = np.maximum(p_values, 1e-10)  # Avoid log(0)

colors_pval = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_values]
bars = ax.bar(range(len(summary)), p_values_safe, color=colors_pval, alpha=0.8)

ax.set_yscale('log')
ax.set_xlabel('Stock', fontsize=14, fontweight='bold')
ax.set_ylabel('P-Value (log scale)', fontsize=14, fontweight='bold')
ax.set_title('Statistical Significance - All 50 Stocks',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels(summary['Ticker'], rotation=45, ha='right')
ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "03_pvalue_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_pvalue_distribution.png")

# Figure 4: Cohen's d Effect Size
fig, ax = plt.subplots(figsize=(20, 10))

cohens_d = summary['Cohen\'s d'].values
colors_cohens = ['#2ecc71' if sig else '#e74c3c' for sig in summary['Significant']]

bars = ax.barh(range(len(summary)), cohens_d, color=colors_cohens, alpha=0.8)

ax.set_yticks(range(len(summary)))
ax.set_yticklabels(summary['Ticker'], fontsize=10)
ax.set_xlabel('Cohen\'s d (Effect Size)', fontsize=14, fontweight='bold')
ax.set_title('Effect Size (Cohen\'s d) - All 50 Stocks',
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Small effect')
ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Medium effect')
ax.axvline(x=0.8, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='Large effect')
ax.axvline(x=-0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=-0.8, color='purple', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='x', alpha=0.3)
ax.legend(fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "04_cohens_d_effect_size.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 04_cohens_d_effect_size.png")

print()

# Step 4: Sector-level analysis
print("[4/5] Creating sector-level analysis visualizations...")

# Figure 5: Sector Summary - AR Comparison
fig, ax = plt.subplots(figsize=(14, 8))

sector_data = sector_summary.reset_index()
x = np.arange(len(sector_data))
width = 0.35

bars1 = ax.bar(x - width/2, sector_data['Avg AR (News)'] * 100, width,
               label='News Days', alpha=0.8, color='#3498db')
bars2 = ax.bar(x + width/2, sector_data['Avg AR (Non-News)'] * 100, width,
               label='Non-News Days', alpha=0.8, color='#95a5a6')

ax.set_xlabel('Sector', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Abnormal Return (%)', fontsize=14, fontweight='bold')
ax.set_title('Average Abnormal Returns by Sector',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(sector_data['sector'], rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "05_sector_ar_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 05_sector_ar_comparison.png")

# Figure 6: Sector Significance Rate
fig, ax = plt.subplots(figsize=(12, 8))

sector_data['Significance_Rate'] = (sector_data['Significant Count'] /
                                     sector_data['Total Stocks'] * 100)

bars = ax.barh(range(len(sector_data)), sector_data['Significance_Rate'],
               color='#3498db', alpha=0.8)

ax.set_yticks(range(len(sector_data)))
ax.set_yticklabels(sector_data['sector'], fontsize=12)
ax.set_xlabel('Significance Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Percentage of Stocks with Significant News Impact by Sector',
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (idx, row) in enumerate(sector_data.iterrows()):
    ax.text(row['Significance_Rate'] + 2, i,
            f"{int(row['Significant Count'])}/{int(row['Total Stocks'])}",
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "06_sector_significance_rate.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 06_sector_significance_rate.png")

# Figure 7: Distribution of Effect Sizes by Sector
fig, ax = plt.subplots(figsize=(14, 8))

sector_order = sector_summary.index.tolist()
results_df['sector_cat'] = pd.Categorical(results_df['sector'],
                                          categories=sector_order,
                                          ordered=True)

sns.boxplot(data=results_df, x='sector_cat', y='difference', ax=ax,
            palette='Set2')
ax.set_xlabel('Sector', fontsize=14, fontweight='bold')
ax.set_ylabel('Difference in AR (News - Non-News)', fontsize=14, fontweight='bold')
ax.set_title('Distribution of News Impact Effect Sizes by Sector',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "07_sector_effect_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 07_sector_effect_distribution.png")

# Figure 8: Scatter - Effect Size vs News Coverage
fig, ax = plt.subplots(figsize=(14, 10))

# Color by sector
sector_colors = plt.cm.tab10(np.linspace(0, 1, len(sector_order)))
sector_color_map = dict(zip(sector_order, sector_colors))

for sector in sector_order:
    sector_data = results_df[results_df['sector'] == sector]
    ax.scatter(sector_data['news_days_pct'],
               sector_data['difference'] * 100,
               label=sector,
               color=sector_color_map[sector],
               alpha=0.7,
               s=100,
               edgecolors='black',
               linewidth=0.5)

ax.set_xlabel('News Coverage (% of Trading Days)', fontsize=14, fontweight='bold')
ax.set_ylabel('Effect Size (Difference in AR, %)', fontsize=14, fontweight='bold')
ax.set_title('News Impact vs News Coverage',
             fontsize=16, fontweight='bold', pad=20)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(SUMMARY_DIR / "08_effect_vs_coverage.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 08_effect_vs_coverage.png")

print()

# Step 5: Generate text summary
print("[5/5] Generating text summary report...")

with open(SUMMARY_DIR / "SUMMARY_REPORT.txt", 'w') as f:
    f.write("="*80 + "\n")
    f.write("EVENT STUDY ANALYSIS - COMPREHENSIVE SUMMARY REPORT\n")
    f.write("All 50 Stocks Across 10 Sectors\n")
    f.write("="*80 + "\n\n")

    # Overall statistics
    f.write("OVERALL STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total Stocks Analyzed: {len(results_df)}\n")
    f.write(f"Stocks with Significant News Impact: {sig_summary['Significant'].sum()} ({sig_summary['Significant'].sum()/len(results_df)*100:.1f}%)\n")
    f.write(f"Stocks with No Significant Impact: {len(results_df) - sig_summary['Significant'].sum()} ({(len(results_df) - sig_summary['Significant'].sum())/len(results_df)*100:.1f}%)\n")
    f.write(f"\nAverage Difference in AR: {results_df['difference'].mean()*100:.4f}%\n")
    f.write(f"Median Difference in AR: {results_df['difference'].median()*100:.4f}%\n")
    f.write(f"Average |Cohen's d|: {results_df['cohens_d'].abs().mean():.3f}\n\n")

    # Positive vs Negative effects
    positive = sig_summary[sig_summary['Difference'] > 0]
    negative = sig_summary[sig_summary['Difference'] < 0]

    f.write("DIRECTION OF SIGNIFICANT EFFECTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Positive News Impact (News days > Non-news days): {len(positive)} stocks\n")
    f.write(f"Negative News Impact (Non-news days > News days): {len(negative)} stocks\n\n")

    # Top performers
    f.write("TOP 10 STRONGEST POSITIVE NEWS IMPACTS\n")
    f.write("-"*80 + "\n")
    top_positive = summary[summary['Significant'] == True].nlargest(10, 'Difference')
    for idx, row in top_positive.iterrows():
        cohens_d = row['Cohen\'s d']
        f.write(f"{row['Ticker']:6} ({row['Sector']:20}): +{row['Difference']*100:.3f}% (p={row['P-Value']:.4f}, d={cohens_d:.3f})\n")

    f.write("\n")
    f.write("TOP 10 STRONGEST NEGATIVE NEWS IMPACTS\n")
    f.write("-"*80 + "\n")
    top_negative = summary[summary['Significant'] == True].nsmallest(10, 'Difference')
    for idx, row in top_negative.iterrows():
        cohens_d = row['Cohen\'s d']
        f.write(f"{row['Ticker']:6} ({row['Sector']:20}): {row['Difference']*100:.3f}% (p={row['P-Value']:.4f}, d={cohens_d:.3f})\n")

    f.write("\n")
    f.write("SECTOR-LEVEL SUMMARY\n")
    f.write("-"*80 + "\n")
    for idx, row in sector_summary.iterrows():
        f.write(f"\n{idx}:\n")
        f.write(f"  Avg AR (News): {row['Avg AR (News)']*100:.3f}%\n")
        f.write(f"  Avg AR (Non-News): {row['Avg AR (Non-News)']*100:.3f}%\n")
        f.write(f"  Avg Difference: {row['Avg Difference']*100:.3f}%\n")
        f.write(f"  Significant: {int(row['Significant Count'])}/{int(row['Total Stocks'])} stocks ({row['Significant Count']/row['Total Stocks']*100:.1f}%)\n")

    f.write("\n")
    f.write("STOCKS WITH NO SIGNIFICANT NEWS IMPACT\n")
    f.write("-"*80 + "\n")
    non_sig = summary[summary['Significant'] == False]
    for idx, row in non_sig.iterrows():
        f.write(f"{row['Ticker']:6} ({row['Sector']:20}): p={row['P-Value']:.4f}\n")

    f.write("\n")
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print("  ✓ Saved: SUMMARY_REPORT.txt")
print()

print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {SUMMARY_DIR}")
print("\nGenerated files:")
print("  • full_summary.csv")
print("  • significant_results.csv")
print("  • sector_summary.csv")
print("  • 01_ar_comparison_all_stocks.png")
print("  • 02_effect_size_all_stocks.png")
print("  • 03_pvalue_distribution.png")
print("  • 04_cohens_d_effect_size.png")
print("  • 05_sector_ar_comparison.png")
print("  • 06_sector_significance_rate.png")
print("  • 07_sector_effect_distribution.png")
print("  • 08_effect_vs_coverage.png")
print("  • SUMMARY_REPORT.txt")
print()

# Print quick summary
print("QUICK SUMMARY:")
print(f"  Total stocks: {len(results_df)}")
print(f"  Significant: {sig_summary['Significant'].sum()} ({sig_summary['Significant'].sum()/len(results_df)*100:.1f}%)")
print(f"  Positive impact: {len(positive)} stocks")
print(f"  Negative impact: {len(negative)} stocks")
print()
