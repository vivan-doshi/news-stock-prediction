"""
CATEGORY × SECTOR AGGREGATION AND VISUALIZATION
================================================

Aggregates category event study results by sector and creates comparative visualizations.

Outputs:
1. Sector-level summaries: by_sector/[SECTOR]/category_summary.csv
2. Sector visualizations: by_sector/[SECTOR]/category_comparison.png
3. Overall category rankings: category_effectiveness_ranking.csv
4. Sector × Category heatmap: sector_category_heatmap.png
5. Comprehensive markdown report: CATEGORY_EVENT_STUDY_REPORT.md

Author: Category Event Study System
Date: 2025-10-13
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import stock configuration
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import importlib
config_module = importlib.import_module('21_expanded_50_stock_config')
STOCKS = config_module.EXPANDED_STOCKS

# Parameters
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "03-output" / "category_event_study"
OUTPUT_DIR = RESULTS_DIR / "by_sector"

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_comprehensive_results() -> pd.DataFrame:
    """Load comprehensive category results"""
    results_file = RESULTS_DIR / "comprehensive_category_results.csv"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}\nPlease run 28_category_event_study.py first")

    df = pd.read_csv(results_file)

    # Filter successful results
    df = df[df['status'] == 'success'].copy()

    print(f"Loaded {len(df)} successful category-stock analyses")

    return df


def aggregate_by_sector(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by sector"""
    print("\n" + "="*80)
    print("AGGREGATING BY SECTOR")
    print("="*80)

    sector_agg = results_df.groupby(['sector', 'category']).agg({
        'ticker': 'count',
        'num_events': 'sum',
        'mean_ar_event': 'mean',
        'mean_ar_non_event': 'mean',
        'std_ar_event': 'mean',
        'p_value_ttest': 'mean',
        'significant': 'sum',
        'cohens_d': 'mean',
        'avg_r_squared': 'mean'
    }).reset_index()

    sector_agg.columns = ['sector', 'category', 'num_stocks', 'total_events',
                          'mean_ar_event', 'mean_ar_non_event', 'mean_std_ar_event',
                          'avg_p_value', 'num_significant', 'avg_cohens_d', 'avg_r_squared']

    sector_agg['ar_difference'] = sector_agg['mean_ar_event'] - sector_agg['mean_ar_non_event']
    sector_agg['pct_significant'] = (sector_agg['num_significant'] / sector_agg['num_stocks'] * 100).round(1)

    return sector_agg


def aggregate_by_category(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by category across all stocks"""
    print("\n" + "="*80)
    print("AGGREGATING BY CATEGORY")
    print("="*80)

    category_agg = results_df.groupby('category').agg({
        'ticker': 'count',
        'num_events': 'sum',
        'mean_ar_event': 'mean',
        'mean_ar_non_event': 'mean',
        'std_ar_event': 'mean',
        'p_value_ttest': 'mean',
        'significant': 'sum',
        'cohens_d': 'mean',
        'avg_r_squared': 'mean'
    }).reset_index()

    category_agg.columns = ['category', 'num_stocks', 'total_events',
                            'mean_ar_event', 'mean_ar_non_event', 'mean_std_ar_event',
                            'avg_p_value', 'num_significant', 'avg_cohens_d', 'avg_r_squared']

    category_agg['ar_difference'] = category_agg['mean_ar_event'] - category_agg['mean_ar_non_event']
    category_agg['pct_significant'] = (category_agg['num_significant'] / category_agg['num_stocks'] * 100).round(1)

    # Sort by effectiveness (combination of effect size and significance)
    category_agg['effectiveness_score'] = (
        np.abs(category_agg['avg_cohens_d']) *
        (category_agg['pct_significant'] / 100)
    )
    category_agg = category_agg.sort_values('effectiveness_score', ascending=False)

    return category_agg


def create_sector_visualizations(sector_agg: pd.DataFrame):
    """Create visualizations for each sector"""
    print("\n" + "="*80)
    print("CREATING SECTOR VISUALIZATIONS")
    print("="*80)

    sectors = sector_agg['sector'].unique()

    for sector in sectors:
        print(f"\n{sector}...")

        sector_dir = OUTPUT_DIR / sector
        sector_dir.mkdir(parents=True, exist_ok=True)

        sector_data = sector_agg[sector_agg['sector'] == sector].copy()

        # Calculate effectiveness score if not present
        if 'effectiveness_score' not in sector_data.columns:
            sector_data['effectiveness_score'] = (
                np.abs(sector_data['avg_cohens_d']) *
                (sector_data['pct_significant'] / 100)
            )

        sector_data = sector_data.sort_values('avg_cohens_d', ascending=False)

        # Save sector summary
        sector_data.to_csv(sector_dir / 'category_summary.csv', index=False)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Mean AR by category
        ax1 = axes[0, 0]
        x = range(len(sector_data))
        width = 0.35
        ax1.bar([i - width/2 for i in x], sector_data['mean_ar_event'], width,
                label='Event Days', color='coral', alpha=0.8)
        ax1.bar([i + width/2 for i in x], sector_data['mean_ar_non_event'], width,
                label='Non-Event Days', color='steelblue', alpha=0.8)
        ax1.set_xlabel('Category', fontweight='bold')
        ax1.set_ylabel('Mean Abnormal Return', fontweight='bold')
        ax1.set_title('Mean AR by News Category', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(sector_data['category'], rotation=45, ha='right')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')

        # 2. Effect size (Cohen's d)
        ax2 = axes[0, 1]
        colors = ['green' if d > 0 else 'red' for d in sector_data['avg_cohens_d']]
        bars = ax2.barh(sector_data['category'], sector_data['avg_cohens_d'], color=colors, alpha=0.7)
        ax2.set_xlabel('Cohen\'s d', fontweight='bold')
        ax2.set_ylabel('Category', fontweight='bold')
        ax2.set_title('Effect Size by Category', fontweight='bold', fontsize=12)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.axvline(0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax2.axvline(-0.2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axvline(-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax2.grid(alpha=0.3, axis='x')

        # 3. Number of events
        ax3 = axes[1, 0]
        ax3.bar(sector_data['category'], sector_data['total_events'], color='steelblue', alpha=0.8)
        ax3.set_xlabel('Category', fontweight='bold')
        ax3.set_ylabel('Total Events', fontweight='bold')
        ax3.set_title('Number of News Events by Category', fontweight='bold', fontsize=12)
        ax3.set_xticklabels(sector_data['category'], rotation=45, ha='right')
        ax3.grid(alpha=0.3, axis='y')

        # 4. Significance percentage
        ax4 = axes[1, 1]
        bars = ax4.bar(sector_data['category'], sector_data['pct_significant'], color='forestgreen', alpha=0.8)
        for i, (bar, n, sig) in enumerate(zip(bars, sector_data['num_stocks'], sector_data['num_significant'])):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(sig)}/{int(n)}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax4.set_xlabel('Category', fontweight='bold')
        ax4.set_ylabel('% Stocks with Significant Results', fontweight='bold')
        ax4.set_title('Statistical Significance by Category', fontweight='bold', fontsize=12)
        ax4.set_xticklabels(sector_data['category'], rotation=45, ha='right')
        ax4.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_ylim([0, 100])
        ax4.grid(alpha=0.3, axis='y')

        plt.suptitle(f'{sector} Sector - Category Comparison', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = sector_dir / 'category_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Saved: {output_file}")


def create_overall_heatmap(sector_agg: pd.DataFrame):
    """Create sector × category heatmap"""
    print("\n" + "="*80)
    print("CREATING OVERALL HEATMAP")
    print("="*80)

    # Pivot: sectors as rows, categories as columns, Cohen's d as values
    heatmap_data = sector_agg.pivot(index='sector', columns='category', values='avg_cohens_d')

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': "Cohen's d (Effect Size)"}, linewidths=0.5,
                linecolor='gray', ax=ax)

    ax.set_title('News Category Impact by Sector\n(Cohen\'s d Effect Size)',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xlabel('News Category', fontweight='bold', fontsize=12)
    ax.set_ylabel('Sector', fontweight='bold', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_file = RESULTS_DIR / 'sector_category_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_file}")


def create_category_ranking_chart(category_agg: pd.DataFrame):
    """Create category effectiveness ranking visualization"""
    print("\n" + "="*80)
    print("CREATING CATEGORY RANKING CHART")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Effectiveness score
    ax1 = axes[0]
    bars = ax1.barh(category_agg['category'], category_agg['effectiveness_score'],
                    color='steelblue', alpha=0.8)
    ax1.set_xlabel('Effectiveness Score\n(|Cohen\'s d| × % Significant)', fontweight='bold')
    ax1.set_ylabel('Category', fontweight='bold')
    ax1.set_title('Category Effectiveness Ranking', fontweight='bold', fontsize=12)
    ax1.grid(alpha=0.3, axis='x')

    # Add values
    for i, (bar, val) in enumerate(zip(bars, category_agg['effectiveness_score'])):
        width = bar.get_width()
        ax1.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

    # 2. Scatter: Effect size vs % significant
    ax2 = axes[1]
    scatter = ax2.scatter(category_agg['avg_cohens_d'], category_agg['pct_significant'],
                         s=category_agg['total_events']/10, alpha=0.7, c=range(len(category_agg)),
                         cmap='viridis', edgecolors='black', linewidth=1)

    for idx, row in category_agg.iterrows():
        ax2.annotate(row['category'], (row['avg_cohens_d'], row['pct_significant']),
                    fontsize=8, ha='center', va='bottom')

    ax2.set_xlabel('Average Cohen\'s d (Effect Size)', fontweight='bold')
    ax2.set_ylabel('% Stocks with Significant Results', fontweight='bold')
    ax2.set_title('Category Performance Map', fontweight='bold', fontsize=12)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(alpha=0.3)

    plt.suptitle('Overall Category Effectiveness Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = RESULTS_DIR / 'category_effectiveness_ranking.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_file}")


def generate_comprehensive_report(results_df: pd.DataFrame, sector_agg: pd.DataFrame,
                                  category_agg: pd.DataFrame):
    """Generate comprehensive markdown report"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*80)

    report = f"""# Category-Specific Event Study Report

**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Methodology:** Fama-French 5-Factor Model with Category-Specific News Events

---

## Executive Summary

This report presents a comprehensive analysis of **how different types of news** impact stock returns across **{len(STOCKS)} stocks** in **{len(sector_agg['sector'].unique())} sectors**.

### Key Findings

**Overall Statistics:**
- **Total Category-Stock Combinations:** {len(results_df[results_df['status'] == 'success'])}
- **Total News Events Analyzed:** {results_df['num_events'].sum():,}
- **Significant Results:** {results_df['significant'].sum()} / {len(results_df)} ({results_df['significant'].sum()/len(results_df)*100:.1f}%)

---

## Category Effectiveness Rankings

### Top 5 Most Impactful News Categories

| Rank | Category | Avg Effect Size (Cohen's d) | % Stocks Significant | Total Events | Effectiveness Score |
|------|----------|------------------------------|---------------------|--------------|---------------------|
"""

    for i, (idx, row) in enumerate(category_agg.head(5).iterrows(), 1):
        report += f"| {i} | {row['category']} | {row['avg_cohens_d']:.3f} | {row['pct_significant']:.1f}% | {int(row['total_events'])} | {row['effectiveness_score']:.3f} |\n"

    report += f"""

### Category Performance Summary

| Category | Avg AR (Event) | Avg AR (Non-Event) | Difference | Avg p-value | Significant Stocks | Effect Size |
|----------|----------------|-------------------|------------|-------------|-------------------|-------------|
"""

    for idx, row in category_agg.iterrows():
        report += f"| {row['category']} | {row['mean_ar_event']:.4f} | {row['mean_ar_non_event']:.4f} | {row['ar_difference']:.4f} | {row['avg_p_value']:.4f} | {int(row['num_significant'])}/{int(row['num_stocks'])} | {row['avg_cohens_d']:.3f} |\n"

    report += f"""

---

## Sector-Level Analysis

### Category Impact by Sector

"""

    for sector in sorted(sector_agg['sector'].unique()):
        sector_data = sector_agg[sector_agg['sector'] == sector].sort_values('avg_cohens_d', ascending=False)

        report += f"""
#### {sector}

**Most Impactful Categories:**
"""
        for idx, row in sector_data.head(3).iterrows():
            impact = "positive" if row['avg_cohens_d'] > 0 else "negative"
            report += f"1. **{row['category']}**: {impact} impact (d={row['avg_cohens_d']:.3f}), {int(row['num_significant'])}/{int(row['num_stocks'])} stocks significant\n"

        report += "\n"

    report += f"""
---

## Key Insights

### 1. Category Effectiveness

**Most Effective Categories:**
{category_agg.head(3)['category'].tolist()}

These categories show the strongest and most consistent impact on stock returns across sectors.

### 2. Sector Sensitivity

Different sectors show varying sensitivity to news categories:

"""

    # Find sector-category combinations with highest effect sizes
    top_combos = sector_agg.nlargest(5, 'avg_cohens_d')[['sector', 'category', 'avg_cohens_d', 'pct_significant']]

    for idx, row in top_combos.iterrows():
        report += f"- **{row['sector']} + {row['category']}**: d={row['avg_cohens_d']:.3f}, {row['pct_significant']:.0f}% significant\n"

    report += f"""

### 3. Statistical Robustness

- Average model fit (R²): {results_df['avg_r_squared'].mean():.3f}
- Median p-value: {results_df['p_value_ttest'].median():.4f}
- Overall significance rate: {results_df['significant'].sum()/len(results_df)*100:.1f}%

---

## Methodology

### News Categories Analyzed

1. **Earnings**: Quarterly/annual earnings reports, guidance updates
2. **Product Launch**: New product announcements, major releases
3. **Executive Changes**: CEO changes, board appointments, executive departures
4. **M&A**: Mergers, acquisitions, strategic partnerships
5. **Regulatory/Legal**: Regulatory approvals/denials, lawsuits, compliance issues
6. **Analyst Ratings**: Analyst upgrades/downgrades, price target changes
7. **Dividends**: Dividend announcements, changes, special dividends
8. **Market Performance**: Milestone achievements, market share changes

### Statistical Methods

1. **Factor Model:** Fama-French 5-Factor (Mkt-RF, SMB, HML, RMW, CMA)
2. **Rolling Window:** 252-day estimation window
3. **Abnormal Returns:** Winsorized at 1% tails
4. **Hypothesis Tests:** Welch's t-test, Mann-Whitney U test
5. **Effect Size:** Cohen's d
6. **Confidence Intervals:** Bootstrap (1000 iterations, 95% CI)

---

## Visualizations

The following visualizations are available:

1. **`sector_category_heatmap.png`** - Overall impact across sectors and categories
2. **`category_effectiveness_ranking.png`** - Category performance comparison
3. **`by_sector/[SECTOR]/category_comparison.png`** - Sector-specific analysis (10 files)
4. **`results/[TICKER]/[CATEGORY]/event_study.png`** - Individual analyses (400 files)

---

## Data Files

### Summary Files
- `comprehensive_category_results.csv` - All {len(results_df)} category-stock results
- `category_summary_statistics.csv` - Aggregated category statistics
- `category_sector_matrix.csv` - Sector × category event counts

### Sector Files
- `by_sector/[SECTOR]/category_summary.csv` - Category statistics per sector (10 files)

### Individual Stock-Category Files
- `results/[TICKER]/[CATEGORY]/summary.csv` - Detailed results (400 files)

---

## Recommendations for Next Phase

Based on this analysis, we recommend:

1. **Focus on High-Impact Categories**: {', '.join(category_agg.head(3)['category'].tolist())}
   - These show strongest and most consistent effects

2. **Sector-Specific Strategies**:
"""

    # Recommendations by sector
    for sector in sorted(sector_agg['sector'].unique()):
        sector_data = sector_agg[sector_agg['sector'] == sector].copy()

        # Calculate effectiveness score if not present
        if 'effectiveness_score' not in sector_data.columns:
            sector_data['effectiveness_score'] = (
                np.abs(sector_data['avg_cohens_d']) *
                (sector_data['pct_significant'] / 100)
            )

        sector_data = sector_data.sort_values('effectiveness_score', ascending=False)
        top_cat = sector_data.iloc[0]
        report += f"   - **{sector}**: Prioritize {top_cat['category']} news (d={top_cat['avg_cohens_d']:.3f})\n"

    report += f"""

3. **Model Refinement**:
   - Consider category-specific event windows (some categories may have delayed/prolonged effects)
   - Investigate sentiment interaction (positive vs negative news within categories)
   - Explore category combinations (multiple simultaneous events)

---

## References

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13-39.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.

---

*Report generated by Category Event Study System*
*Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Save report
    report_file = RESULTS_DIR / "CATEGORY_EVENT_STUDY_REPORT.md"
    with open(report_file, 'w') as f:
        f.write(report)

    print(f"✅ Saved: {report_file}")


def main():
    """Main aggregation and visualization function"""
    print("="*80)
    print("CATEGORY × SECTOR AGGREGATION")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    results_df = load_comprehensive_results()

    # Aggregate by sector and category
    sector_agg = aggregate_by_sector(results_df)
    category_agg = aggregate_by_category(results_df)

    # Save aggregated results
    sector_agg.to_csv(RESULTS_DIR / 'sector_category_aggregated.csv', index=False)
    category_agg.to_csv(RESULTS_DIR / 'category_effectiveness_ranking.csv', index=False)

    print("\n" + "="*80)
    print("CATEGORY RANKINGS")
    print("="*80)
    print(category_agg[['category', 'avg_cohens_d', 'pct_significant', 'total_events',
                        'effectiveness_score']].to_string(index=False))

    # Create visualizations
    create_sector_visualizations(sector_agg)
    create_overall_heatmap(sector_agg)
    create_category_ranking_chart(category_agg)

    # Generate report
    generate_comprehensive_report(results_df, sector_agg, category_agg)

    print("\n" + "="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"\n📁 Sector visualizations: {OUTPUT_DIR}")
    print(f"📊 Overall visualizations: {RESULTS_DIR}")
    print(f"📄 Comprehensive report: {RESULTS_DIR / 'CATEGORY_EVENT_STUDY_REPORT.md'}")

    return sector_agg, category_agg


if __name__ == "__main__":
    sector_agg, category_agg = main()