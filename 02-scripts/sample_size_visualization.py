"""
Sample Size Visualization: Event Days vs Non-Event Days by Stock
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Read results
results_file = Path(__file__).parent.parent / "03-output/balanced_event_study/results_summary.csv"
df = pd.read_csv(results_file)

# Calculate percentages
df['news_days_pct'] = (df['news_days'] / df['total_days']) * 100
df['non_news_days_pct'] = (df['non_news_days'] / df['total_days']) * 100

# Create output directory
output_dir = Path(__file__).parent.parent / "03-output/balanced_event_study"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Create comprehensive visualization
fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(4, 2, hspace=0.6, wspace=0.35, top=0.96, bottom=0.04, left=0.06, right=0.98)

# 1. Stacked bar chart - All stocks
ax1 = fig.add_subplot(gs[0:2, 0])
x_pos = range(len(df))
ax1.barh(x_pos, df['news_days'], color='coral', alpha=0.8, label='News Days (Event)')
ax1.barh(x_pos, df['non_news_days'], left=df['news_days'], color='steelblue', alpha=0.8, label='Non-News Days (Non-Event)')

ax1.set_yticks(x_pos)
ax1.set_yticklabels(df['ticker'], fontsize=9)
ax1.set_xlabel('Number of Days', fontweight='bold', fontsize=11)
ax1.set_ylabel('Stock Ticker', fontweight='bold', fontsize=11)
ax1.set_title('Sample Size Distribution: Event Days vs Non-Event Days (All 50 Stocks)',
              fontweight='bold', fontsize=13)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(alpha=0.3, axis='x')

# Add percentage labels
for i, (idx, row) in enumerate(df.iterrows()):
    ax1.text(row['total_days']/2, i, f"{row['news_days_pct']:.1f}% | {row['non_news_days_pct']:.1f}%",
             ha='center', va='center', fontsize=7, fontweight='bold', color='black')

# 2. Percentage stacked bar chart by sector
ax2 = fig.add_subplot(gs[0, 1])
sector_summary = df.groupby('sector').agg({
    'news_days': 'sum',
    'non_news_days': 'sum',
    'total_days': 'sum'
}).reset_index()
sector_summary['news_days_pct'] = (sector_summary['news_days'] / sector_summary['total_days']) * 100
sector_summary['non_news_days_pct'] = (sector_summary['non_news_days'] / sector_summary['total_days']) * 100

x_pos = range(len(sector_summary))
width = 0.8
ax2.bar(x_pos, sector_summary['news_days_pct'], width, color='coral', alpha=0.8, label='News Days %')
ax2.bar(x_pos, sector_summary['non_news_days_pct'], width, bottom=sector_summary['news_days_pct'],
        color='steelblue', alpha=0.8, label='Non-News Days %')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(sector_summary['sector'], rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Percentage (%)', fontweight='bold')
ax2.set_title('Event vs Non-Event Days by Sector', fontweight='bold', fontsize=12)
ax2.legend(loc='upper right', fontsize=9)
ax2.set_ylim([0, 100])
ax2.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.3)
ax2.grid(alpha=0.3, axis='y')

# Add percentage labels
for i, row in sector_summary.iterrows():
    ax2.text(i, row['news_days_pct']/2, f"{row['news_days_pct']:.1f}%",
             ha='center', va='center', fontsize=8, fontweight='bold', color='black')
    ax2.text(i, row['news_days_pct'] + row['non_news_days_pct']/2, f"{row['non_news_days_pct']:.1f}%",
             ha='center', va='center', fontsize=8, fontweight='bold', color='white')

# 3. Distribution histogram
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(df['news_days_pct'], bins=20, color='coral', alpha=0.7, edgecolor='black', linewidth=1)
ax3.axvline(df['news_days_pct'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df["news_days_pct"].mean():.1f}%')
ax3.axvline(50, color='black', linestyle=':', linewidth=2, alpha=0.5, label='50% threshold')
ax3.set_xlabel('Event Days (%)', fontweight='bold')
ax3.set_ylabel('Number of Stocks', fontweight='bold')
ax3.set_title('Distribution of Event Day Percentage Across Stocks', fontweight='bold', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, axis='y')

# 4. Summary table
ax4 = fig.add_subplot(gs[2:4, :])
ax4.axis('off')

summary_stats = f"""
{'='*140}
SAMPLE SIZE SUMMARY STATISTICS
{'='*140}

Overall Statistics:
  • Total Stocks: {len(df)}
  • Total Trading Days: {df['total_days'].sum():,}
  • Total Event Days: {df['news_days'].sum():,} ({df['news_days'].sum()/df['total_days'].sum()*100:.2f}%)
  • Total Non-Event Days: {df['non_news_days'].sum():,} ({df['non_news_days'].sum()/df['total_days'].sum()*100:.2f}%)

Event Day Percentage Statistics:
  • Mean: {df['news_days_pct'].mean():.2f}%
  • Median: {df['news_days_pct'].median():.2f}%
  • Std Dev: {df['news_days_pct'].std():.2f}%
  • Min: {df['news_days_pct'].min():.2f}% ({df.loc[df['news_days_pct'].idxmin(), 'ticker']})
  • Max: {df['news_days_pct'].max():.2f}% ({df.loc[df['news_days_pct'].idxmax(), 'ticker']})

Stocks by Event Day Coverage:
  • High (>60%): {(df['news_days_pct'] > 60).sum()} stocks
  • Medium (40-60%): {((df['news_days_pct'] >= 40) & (df['news_days_pct'] <= 60)).sum()} stocks
  • Low (<40%): {(df['news_days_pct'] < 40).sum()} stocks
{'='*140}
"""

ax4.text(0.05, 0.7, summary_stats, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Event Study Sample Size Analysis: Event Days vs Non-Event Days (50 Stocks)',
             fontsize=17, fontweight='bold', y=0.985)

# Save
output_file = output_dir / 'sample_size_distribution.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_file}")
plt.close()

# Create detailed percentage table
print("\n" + "="*100)
print("DETAILED PERCENTAGE BREAKDOWN BY STOCK")
print("="*100)
print(f"\n{'Ticker':<8} {'Sector':<25} {'Total':<8} {'Event Days':<15} {'Non-Event Days':<15} {'Event %':<10} {'Non-Event %':<12}")
print("-"*100)

for idx, row in df.iterrows():
    print(f"{row['ticker']:<8} {row['sector']:<25} {int(row['total_days']):<8} "
          f"{int(row['news_days']):<15} {int(row['non_news_days']):<15} "
          f"{row['news_days_pct']:>7.2f}%   {row['non_news_days_pct']:>7.2f}%")

print("-"*100)
print(f"{'TOTAL':<8} {'':<25} {int(df['total_days'].sum()):<8} "
      f"{int(df['news_days'].sum()):<15} {int(df['non_news_days'].sum()):<15} "
      f"{df['news_days'].sum()/df['total_days'].sum()*100:>7.2f}%   "
      f"{df['non_news_days'].sum()/df['total_days'].sum()*100:>7.2f}%")
print("="*100)

# Save percentage table to CSV
pct_table = df[['ticker', 'sector', 'total_days', 'news_days', 'non_news_days',
                'news_days_pct', 'non_news_days_pct']].copy()
pct_table.columns = ['Ticker', 'Sector', 'Total_Days', 'Event_Days', 'Non_Event_Days',
                     'Event_Days_%', 'Non_Event_Days_%']
pct_table.to_csv(output_dir / 'sample_size_percentages.csv', index=False, float_format='%.2f')
print(f"\n✅ Saved: {output_dir / 'sample_size_percentages.csv'}")

# Create scatter plot with ticker names instead of bubbles
print("\n" + "="*100)
print("CREATING SCATTER PLOT")
print("="*100)

fig_scatter = plt.figure(figsize=(32, 20))

# Get unique sectors and assign colors
sectors = df['sector'].unique()
color_palette = sns.color_palette("tab10", n_colors=len(sectors))
sector_colors = dict(zip(sectors, color_palette))

# Create scatter plot
ax = fig_scatter.add_subplot(111)

# Plot ticker names directly as text with no overlap adjustment
for idx, row in df.iterrows():
    ax.text(row['news_days'], row['news_days_pct'], row['ticker'],
            fontsize=14,
            fontweight='bold',
            color='black',
            ha='center',
            va='center',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=sector_colors[row['sector']],
                     alpha=0.9, edgecolor='black', linewidth=2.5),
            zorder=3)

# Create invisible scatter points for legend (one per sector)
legend_handles = []
for sector in sectors:
    sector_data = df[df['sector'] == sector]
    handle = ax.scatter([], [], s=250, alpha=0.9, c=[sector_colors[sector]],
                       edgecolors='black', linewidth=2, label=sector)
    legend_handles.append(handle)

# Add reference lines
ax.axhline(50, color='gray', linestyle='--', linewidth=2.5, alpha=0.6, label='50% threshold', zorder=1)
ax.axhline(df['news_days_pct'].mean(), color='red', linestyle='--', linewidth=2.5, alpha=0.7,
          label=f'Mean: {df["news_days_pct"].mean():.1f}%', zorder=1)

# Formatting
ax.set_xlabel('Number of Event Days', fontweight='bold', fontsize=16)
ax.set_ylabel('Event Days (%)', fontweight='bold', fontsize=16)
ax.set_title('Sample Size Analysis: Event Days Count vs Event Days Percentage by Stock and Sector',
            fontweight='bold', fontsize=18, pad=20)
ax.legend(loc='upper left', fontsize=12, framealpha=0.95, ncol=2,
          edgecolor='black', fancybox=True, shadow=True)
ax.grid(alpha=0.4, linestyle=':', linewidth=1, zorder=0)
ax.set_ylim([df['news_days_pct'].min() - 8, df['news_days_pct'].max() + 8])
ax.set_xlim([df['news_days'].min() - 100, df['news_days'].max() + 100])

# Add summary statistics box
stats_text = f"""Summary Statistics:
Total Stocks: {len(df)}
Mean Event Days: {df['news_days'].mean():.0f}
Mean Event %: {df['news_days_pct'].mean():.1f}%
Median Event %: {df['news_days_pct'].median():.1f}%
Std Dev: {df['news_days_pct'].std():.1f}%"""

ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2),
        fontfamily='monospace', zorder=5)

plt.tight_layout()
scatter_file = output_dir / 'sample_size_scatter.png'
plt.savefig(scatter_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {scatter_file}")
plt.close()

print("\n" + "="*100)
print("CREATING RANKED BAR CHART BY SECTOR")
print("="*100)

# Get unique sectors and assign colors
sectors = df['sector'].unique()
color_palette = sns.color_palette("tab10", n_colors=len(sectors))
sector_colors = dict(zip(sectors, color_palette))

# Create faceted plot - one subplot per sector
n_sectors = len(sectors)
fig_facet = plt.figure(figsize=(24, 32))
gs = fig_facet.add_gridspec(5, 2, hspace=0.4, wspace=0.3, top=0.97, bottom=0.03, left=0.08, right=0.98)

sector_positions = [
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
    (2, 1), (3, 0), (3, 1), (4, 0), (4, 1)
]

for idx, sector in enumerate(sorted(sectors)):
    row, col = sector_positions[idx]
    ax = fig_facet.add_subplot(gs[row, col])

    # Filter data for this sector and sort
    sector_df = df[df['sector'] == sector].sort_values('news_days_pct', ascending=True).reset_index(drop=True)

    # Create horizontal bars
    y_pos = range(len(sector_df))
    bars = ax.barh(y_pos, sector_df['news_days_pct'],
                   color=sector_colors[sector], alpha=0.85, edgecolor='black', linewidth=1.5)

    # Add ticker labels on the left
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sector_df['ticker'], fontsize=10, fontweight='bold')

    # Add value annotations
    for i, (_, row_data) in enumerate(sector_df.iterrows()):
        # Percentage inside bar
        if row_data['news_days_pct'] > 15:
            ax.text(row_data['news_days_pct'] - 2, i, f"{row_data['news_days_pct']:.1f}%",
                   ha='right', va='center', fontsize=9, fontweight='bold', color='white')
        else:
            ax.text(row_data['news_days_pct'] + 1, i, f"{row_data['news_days_pct']:.1f}%",
                   ha='left', va='center', fontsize=9, fontweight='bold', color='black')
        # Count outside bar
        ax.text(row_data['news_days_pct'] + 1, i, f"n={int(row_data['news_days'])}",
               ha='left', va='center', fontsize=8, color='gray')

    # Add reference lines
    ax.axvline(50, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=0)
    ax.axvline(df['news_days_pct'].mean(), color='red', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)

    # Formatting
    ax.set_xlabel('Event Days %', fontweight='bold', fontsize=11)
    ax.set_title(f'{sector} (n={len(sector_df)} stocks)', fontweight='bold', fontsize=13,
                pad=10, color=sector_colors[sector])
    ax.set_xlim([0, 100])
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.8, axis='x', zorder=0)

    # Add sector statistics
    sector_mean = sector_df['news_days_pct'].mean()
    sector_median = sector_df['news_days_pct'].median()
    stats_text = f"μ={sector_mean:.1f}% | M={sector_median:.1f}%"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                    edgecolor=sector_colors[sector], linewidth=2),
           fontfamily='monospace')

plt.suptitle('Event Days Coverage by Sector: All 50 Stocks Ranked Within Sectors\n(Gray line = 50%, Red line = Overall Mean 55.3%)',
            fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout()
facet_file = output_dir / 'sample_size_by_sector.png'
plt.savefig(facet_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {facet_file}")
plt.close()

print("\n" + "="*100)
print("ALL VISUALIZATIONS COMPLETE")
print("="*100)