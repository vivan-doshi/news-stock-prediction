"""
Sector-Level Analysis with Comprehensive Visualizations
=======================================================

This script:
1. Creates sector-level folders
2. Generates comparative visualizations for each sector
3. Creates detailed summary tables for each sector
4. Produces sector-specific reports
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
SECTOR_DIR = OUTPUT_DIR / "sector_analysis"
SECTOR_DIR.mkdir(exist_ok=True)

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
print("SECTOR-LEVEL ANALYSIS WITH COMPREHENSIVE VISUALIZATIONS")
print("="*80)
print()

# Step 1: Collect all stock results
print("[1/4] Loading data from all stocks...")
all_data = []

for ticker, sector in SECTOR_MAPPING.items():
    summary_file = OUTPUT_DIR / ticker / "summary.csv"
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df['ticker'] = ticker
        df['sector'] = sector
        all_data.append(df)
        print(f"  ✓ {ticker} ({sector})")

results_df = pd.concat(all_data, ignore_index=True)
print(f"\nLoaded data from {len(all_data)} stocks across {results_df['sector'].nunique()} sectors")
print()

# Calculate derived columns
results_df['difference'] = results_df['mean_ar_news'] - results_df['mean_ar_non_news']
results_df['news_days_pct'] = (results_df['news_days'] / results_df['total_days'] * 100)
results_df['significant'] = results_df['significant_ttest']

# Step 2: Process each sector
print("[2/4] Creating sector-level analysis...")

sectors = results_df['sector'].unique()

for sector in sorted(sectors):
    print(f"\n  Processing {sector}...")

    # Create sector folder
    sector_folder = SECTOR_DIR / sector.replace(" ", "_")
    sector_folder.mkdir(exist_ok=True)

    # Filter data for this sector
    sector_data = results_df[results_df['sector'] == sector].copy()
    sector_data = sector_data.sort_values('difference', ascending=False)

    # ===== 1. Sector Summary Table =====
    summary_table = sector_data[[
        'ticker', 'mean_ar_news', 'mean_ar_non_news', 'difference',
        'p_value_ttest', 'significant', 'cohens_d', 'news_days_pct'
    ]].copy()

    summary_table.columns = [
        'Ticker', 'AR (News)', 'AR (Non-News)', 'Difference',
        'P-Value', 'Significant', 'Cohen\'s d', 'News Coverage %'
    ]

    summary_table.to_csv(sector_folder / "sector_summary.csv", index=False)
    print(f"    ✓ sector_summary.csv")

    # ===== 2. AR Comparison Chart =====
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(sector_data))
    width = 0.35

    bars1 = ax.bar(x - width/2, sector_data['mean_ar_news'] * 100, width,
                   label='News Days', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, sector_data['mean_ar_non_news'] * 100, width,
                   label='Non-News Days', alpha=0.8, color='#95a5a6')

    ax.set_xlabel('Stock', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Abnormal Return (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{sector} - Abnormal Returns Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(sector_data['ticker'], fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(sector_folder / "01_ar_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 01_ar_comparison.png")

    # ===== 3. Effect Size Chart =====
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#2ecc71' if sig else '#e74c3c' for sig in sector_data['significant']]
    bars = ax.barh(range(len(sector_data)), sector_data['difference'] * 100,
                   color=colors, alpha=0.8)

    ax.set_yticks(range(len(sector_data)))
    ax.set_yticklabels(sector_data['ticker'], fontsize=12)
    ax.set_xlabel('Difference in AR (News - Non-News) (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{sector} - News Impact Effect Size',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.8, label='Significant (p < 0.05)'),
        Patch(facecolor='#e74c3c', alpha=0.8, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, fontsize=12)

    plt.tight_layout()
    plt.savefig(sector_folder / "02_effect_size.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 02_effect_size.png")

    # ===== 4. Statistical Significance Chart =====
    fig, ax = plt.subplots(figsize=(12, 8))

    p_values = sector_data['p_value_ttest'].values
    p_values_safe = np.maximum(p_values, 1e-10)

    colors_pval = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_values]
    bars = ax.bar(range(len(sector_data)), p_values_safe, color=colors_pval, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('Stock', fontsize=14, fontweight='bold')
    ax.set_ylabel('P-Value (log scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'{sector} - Statistical Significance',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(sector_data)))
    ax.set_xticklabels(sector_data['ticker'], fontsize=12)
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(sector_folder / "03_statistical_significance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 03_statistical_significance.png")

    # ===== 5. Cohen's d Effect Size =====
    fig, ax = plt.subplots(figsize=(12, 8))

    cohens_d = sector_data['cohens_d'].values
    colors_cohens = ['#2ecc71' if sig else '#e74c3c' for sig in sector_data['significant']]

    bars = ax.barh(range(len(sector_data)), cohens_d, color=colors_cohens, alpha=0.8)

    ax.set_yticks(range(len(sector_data)))
    ax.set_yticklabels(sector_data['ticker'], fontsize=12)
    ax.set_xlabel('Cohen\'s d (Effect Size)', fontsize=14, fontweight='bold')
    ax.set_title(f'{sector} - Effect Size (Cohen\'s d)',
                 fontsize=16, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.2)')
    ax.axvline(x=0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.5)')
    ax.axvline(x=0.8, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='Large (0.8)')
    ax.axvline(x=-0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.5, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-0.8, color='purple', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')

    plt.tight_layout()
    plt.savefig(sector_folder / "04_cohens_d.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 04_cohens_d.png")

    # ===== 6. Comprehensive 4-Panel Visualization =====
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: AR Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sector_data))
    width = 0.35
    ax1.bar(x - width/2, sector_data['mean_ar_news'] * 100, width,
            label='News Days', alpha=0.8, color='#3498db')
    ax1.bar(x + width/2, sector_data['mean_ar_non_news'] * 100, width,
            label='Non-News Days', alpha=0.8, color='#95a5a6')
    ax1.set_xlabel('Stock', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg AR (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Average Abnormal Returns', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sector_data['ticker'], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Panel 2: Effect Size
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#2ecc71' if sig else '#e74c3c' for sig in sector_data['significant']]
    ax2.barh(range(len(sector_data)), sector_data['difference'] * 100,
             color=colors, alpha=0.8)
    ax2.set_yticks(range(len(sector_data)))
    ax2.set_yticklabels(sector_data['ticker'], fontsize=10)
    ax2.set_xlabel('Difference (%)', fontsize=12, fontweight='bold')
    ax2.set_title('News Impact Effect Size', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)

    # Panel 3: P-Values
    ax3 = fig.add_subplot(gs[1, 0])
    p_values = sector_data['p_value_ttest'].values
    p_values_safe = np.maximum(p_values, 1e-10)
    colors_pval = ['#2ecc71' if p < 0.05 else '#e74c3c' for p in p_values]
    ax3.bar(range(len(sector_data)), p_values_safe, color=colors_pval, alpha=0.8)
    ax3.set_yscale('log')
    ax3.set_xlabel('Stock', fontsize=12, fontweight='bold')
    ax3.set_ylabel('P-Value (log)', fontsize=12, fontweight='bold')
    ax3.set_title('Statistical Significance', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(sector_data)))
    ax3.set_xticklabels(sector_data['ticker'], fontsize=10)
    ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=2)
    ax3.grid(axis='y', alpha=0.3, which='both')

    # Panel 4: Cohen's d
    ax4 = fig.add_subplot(gs[1, 1])
    cohens_d = sector_data['cohens_d'].values
    colors_cohens = ['#2ecc71' if sig else '#e74c3c' for sig in sector_data['significant']]
    ax4.barh(range(len(sector_data)), cohens_d, color=colors_cohens, alpha=0.8)
    ax4.set_yticks(range(len(sector_data)))
    ax4.set_yticklabels(sector_data['ticker'], fontsize=10)
    ax4.set_xlabel('Cohen\'s d', fontsize=12, fontweight='bold')
    ax4.set_title('Effect Size (Cohen\'s d)', fontsize=14, fontweight='bold')
    ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax4.axvline(x=0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax4.axvline(x=-0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax4.grid(axis='x', alpha=0.3)

    fig.suptitle(f'{sector} Sector - Comprehensive Analysis',
                 fontsize=18, fontweight='bold', y=0.995)

    plt.savefig(sector_folder / "05_comprehensive_4panel.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 05_comprehensive_4panel.png")

    # ===== 7. Sector Report =====
    with open(sector_folder / "SECTOR_REPORT.txt", 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"{sector.upper()} SECTOR - DETAILED ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write("SECTOR STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Stocks: {len(sector_data)}\n")
        f.write(f"Significant Results: {sector_data['significant'].sum()} ({sector_data['significant'].sum()/len(sector_data)*100:.1f}%)\n")
        f.write(f"Non-Significant: {(~sector_data['significant']).sum()} ({(~sector_data['significant']).sum()/len(sector_data)*100:.1f}%)\n")
        f.write(f"\nAverage AR (News): {sector_data['mean_ar_news'].mean()*100:.3f}%\n")
        f.write(f"Average AR (Non-News): {sector_data['mean_ar_non_news'].mean()*100:.3f}%\n")
        f.write(f"Average Difference: {sector_data['difference'].mean()*100:.3f}%\n")
        f.write(f"Average |Cohen's d|: {sector_data['cohens_d'].abs().mean():.3f}\n\n")

        f.write("STOCK-BY-STOCK RESULTS\n")
        f.write("-"*80 + "\n")
        for _, row in sector_data.iterrows():
            sig_marker = "✓" if row['significant'] else "✗"
            f.write(f"\n{row['ticker']}: {sig_marker}\n")
            f.write(f"  AR (News): {row['mean_ar_news']*100:.3f}%\n")
            f.write(f"  AR (Non-News): {row['mean_ar_non_news']*100:.3f}%\n")
            f.write(f"  Difference: {row['difference']*100:.3f}%\n")
            f.write(f"  P-Value: {row['p_value_ttest']:.4f}\n")
            f.write(f"  Cohen's d: {row['cohens_d']:.3f}\n")
            f.write(f"  News Coverage: {row['news_days_pct']:.1f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"    ✓ SECTOR_REPORT.txt")

print()

# Step 3: Create sector comparison visualizations
print("[3/4] Creating cross-sector comparison visualizations...")

# Aggregate by sector
sector_agg = results_df.groupby('sector').agg({
    'mean_ar_news': 'mean',
    'mean_ar_non_news': 'mean',
    'difference': 'mean',
    'significant': 'sum',
    'ticker': 'count',
    'cohens_d': lambda x: x.abs().mean()
}).round(4)

sector_agg.columns = ['Avg AR (News)', 'Avg AR (Non-News)', 'Avg Difference',
                      'Significant Count', 'Total Stocks', 'Avg |Cohen\'s d|']
sector_agg['Significance Rate (%)'] = (sector_agg['Significant Count'] /
                                       sector_agg['Total Stocks'] * 100)
sector_agg = sector_agg.sort_values('Avg Difference', ascending=False)

# Save sector comparison table
sector_agg.to_csv(SECTOR_DIR / "sector_comparison.csv")
print("  ✓ sector_comparison.csv")

print()

# Step 4: Create README
print("[4/4] Creating sector analysis README...")

with open(SECTOR_DIR / "README.md", 'w') as f:
    f.write("# Sector-Level Event Study Analysis\n\n")
    f.write("## Overview\n\n")
    f.write(f"This folder contains detailed sector-by-sector analysis for {results_df['sector'].nunique()} sectors.\n\n")
    f.write("## Sector Folders\n\n")
    f.write("Each sector has its own folder with:\n\n")
    f.write("1. **sector_summary.csv** - Detailed results for all stocks in the sector\n")
    f.write("2. **01_ar_comparison.png** - Bar chart comparing AR on news vs non-news days\n")
    f.write("3. **02_effect_size.png** - Effect size (difference) for each stock\n")
    f.write("4. **03_statistical_significance.png** - P-value distribution\n")
    f.write("5. **04_cohens_d.png** - Cohen's d effect sizes with benchmarks\n")
    f.write("6. **05_comprehensive_4panel.png** - Combined 4-panel visualization\n")
    f.write("7. **SECTOR_REPORT.txt** - Detailed text report\n\n")
    f.write("## Sectors Analyzed\n\n")

    for sector in sorted(sectors):
        folder_name = sector.replace(" ", "_")
        count = len(results_df[results_df['sector'] == sector])
        sig_count = results_df[results_df['sector'] == sector]['significant'].sum()
        f.write(f"- **[{sector}]({folder_name}/)** - {count} stocks ({sig_count} significant)\n")

    f.write("\n## Sector Comparison\n\n")
    f.write("See [sector_comparison.csv](sector_comparison.csv) for aggregated statistics across all sectors.\n")

print("  ✓ README.md")

print()
print("="*80)
print("SECTOR ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved to: {SECTOR_DIR}")
print(f"Total sectors analyzed: {results_df['sector'].nunique()}")
print(f"Total stock-sector folders created: {len(sectors)}")
print()