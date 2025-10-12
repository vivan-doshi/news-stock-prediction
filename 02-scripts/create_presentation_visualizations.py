"""
Create Professional Visualizations for Stakeholder Presentation
Generates comprehensive charts for event study analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PresentationVisualizer:
    """Creates professional visualizations for stakeholder presentation"""
    
    def __init__(self, output_dir="../03-output/presentation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set figure parameters for high quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['axes.labelsize'] = 20
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        plt.rcParams['legend.fontsize'] = 19
        plt.rcParams['figure.autolayout'] = True
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['axes.labelpad'] = 20
        plt.rcParams['figure.subplot.hspace'] = 0.4
        plt.rcParams['figure.subplot.wspace'] = 0.4
        
    def load_data(self, ticker):
        """Load analysis data for a ticker"""
        data_dir = Path("../03-output")
        
        # Load abnormal returns
        ar_file = data_dir / f"{ticker}_major_events" / "abnormal_returns.csv"
        if ar_file.exists():
            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
        else:
            return None, None, None
            
        # Load beta estimates
        beta_file = data_dir / f"{ticker}_major_events" / "beta_estimates.csv"
        if beta_file.exists():
            beta_df = pd.read_csv(beta_file, index_col=0, parse_dates=True)
        else:
            beta_df = None
            
        # Load analysis summary
        summary_file = data_dir / f"{ticker}_major_events" / "analysis_summary.csv"
        if summary_file.exists():
            summary_df = pd.read_csv(summary_file)
        else:
            summary_df = None
            
        return ar_df, beta_df, summary_df
    
    def create_overview_dashboard(self, tickers=['AAPL', 'TSLA']):
        """Create executive summary dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(30, 24))
        fig.suptitle('Event Study Analysis - Executive Summary', fontsize=36, fontweight='bold', y=0.98)
        
        # Colors for each ticker
        colors = {'AAPL': '#1f77b4', 'TSLA': '#ff7f0e'}
        
        # Data storage for summary
        summary_data = []
        
        for i, ticker in enumerate(tickers):
            ar_df, beta_df, summary_df = self.load_data(ticker)
            if ar_df is None:
                continue
                
            # Calculate summary statistics
            news_days = ar_df[ar_df['News_Day'] == True]
            non_news_days = ar_df[ar_df['News_Day'] == False]
            
            if len(news_days) > 0 and len(non_news_days) > 0:
                news_ar = news_days['Abnormal_Return'].dropna()
                non_news_ar = non_news_days['Abnormal_Return'].dropna()
                
                impact = (news_ar.mean() - non_news_ar.mean()) * 100
                total_days = len(ar_df)
                event_frequency = len(news_days) / total_days * 100
                
                summary_data.append({
                    'Ticker': ticker,
                    'Event Days': len(news_days),
                    'Total Days': total_days,
                    'Event Frequency': event_frequency,
                    'Mean AR (Events)': news_ar.mean() * 100,
                    'Mean AR (Non-Events)': non_news_ar.mean() * 100,
                    'Impact': impact,
                    'Color': colors[ticker]
                })
        
        # 1. Event Frequency Bar Chart
        ax1 = axes[0, 0]
        tickers_list = [d['Ticker'] for d in summary_data]
        frequencies = [d['Event Frequency'] for d in summary_data]
        colors_list = [d['Color'] for d in summary_data]
        
        bars = ax1.bar(tickers_list, frequencies, color=colors_list, alpha=0.7)
        ax1.set_title('Event Frequency', fontweight='bold', fontsize=26)
        ax1.set_ylabel('Event Days (%)', fontsize=22)
        ax1.set_ylim(0, max(frequencies) * 1.5)
        
        # Add value labels on bars
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{freq:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Impact Magnitude Comparison
        ax2 = axes[0, 1]
        impacts = [d['Impact'] for d in summary_data]
        colors_list = [d['Color'] for d in summary_data]
        
        bars = ax2.bar(tickers_list, impacts, color=colors_list, alpha=0.7)
        ax2.set_title('Economic Impact Magnitude', fontweight='bold', fontsize=26)
        ax2.set_ylabel('Impact (%)', fontsize=22)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% Threshold')
        ax2.axhline(y=-1, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, impact in zip(bars, impacts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                    f'{impact:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                    fontweight='bold', color='red' if abs(impact) > 1 else 'black')
        
        ax2.legend()
        
        # 3. Model Quality (R²)
        ax3 = axes[0, 2]
        r_squared_values = []
        for ticker in tickers_list:
            _, beta_df, _ = self.load_data(ticker)
            if beta_df is not None:
                r2 = beta_df['R_squared'].mean()
                r_squared_values.append(r2)
            else:
                r_squared_values.append(0)
        
        bars = ax3.bar(tickers_list, r_squared_values, color=colors_list, alpha=0.7)
        ax3.set_title('Model Quality (R²)', fontweight='bold', fontsize=26)
        ax3.set_ylabel('R²', fontsize=22)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, r2 in zip(bars, r_squared_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Abnormal Returns Distribution
        ax4 = axes[1, 0]
        for i, ticker in enumerate(tickers_list):
            ar_df, _, _ = self.load_data(ticker)
            if ar_df is not None:
                ar_data = ar_df['Abnormal_Return'].dropna() * 100
                ax4.hist(ar_data, bins=30, alpha=0.6, label=ticker, 
                        color=colors_list[i], density=True)
        
        ax4.set_title('Abnormal Returns Distribution', fontweight='bold', fontsize=26)
        ax4.set_xlabel('Abnormal Return (%)', fontsize=22)
        ax4.set_ylabel('Density', fontsize=22)
        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax4.legend(fontsize=20)
        
        # 5. News vs Non-News Comparison
        ax5 = axes[1, 1]
        news_means = [d['Mean AR (Events)'] for d in summary_data]
        non_news_means = [d['Mean AR (Non-Events)'] for d in summary_data]
        
        x = np.arange(len(tickers_list))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, news_means, width, label='Event Days', 
                       color=[c for c in colors_list], alpha=0.7)
        bars2 = ax5.bar(x + width/2, non_news_means, width, label='Non-Event Days', 
                       color=[c for c in colors_list], alpha=0.4)
        
        ax5.set_title('Mean Abnormal Returns Comparison', fontweight='bold', fontsize=26)
        ax5.set_ylabel('Mean AR (%)', fontsize=22)
        ax5.set_xticks(x)
        ax5.set_xticklabels(tickers_list, fontsize=20)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.legend(fontsize=20)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.03),
                        f'{height:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=9)
        
        # 6. Success Criteria Assessment
        ax6 = axes[1, 2]
        criteria_labels = ['Statistical\nSignificance', 'Economic\nImpact (>1%)', 'Model\nQuality (R²>0.4)']
        
        # Create success matrix
        success_matrix = []
        for ticker in tickers_list:
            data = next((d for d in summary_data if d['Ticker'] == ticker), None)
            if data:
                # Statistical significance (simplified - would need p-values)
                stat_sig = abs(data['Impact']) > 0.5  # Simplified threshold
                # Economic impact
                econ_impact = abs(data['Impact']) > 1.0
                # Model quality
                _, beta_df, _ = self.load_data(ticker)
                model_quality = beta_df['R_squared'].mean() > 0.4 if beta_df is not None else False
                
                success_matrix.append([stat_sig, econ_impact, model_quality])
        
        # Plot success matrix
        success_array = np.array(success_matrix)
        im = ax6.imshow(success_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax6.set_title('Success Criteria Assessment', fontweight='bold', fontsize=26)
        ax6.set_xticks(range(len(criteria_labels)))
        ax6.set_xticklabels(criteria_labels, rotation=45, ha='right', fontsize=18)
        ax6.set_yticks(range(len(tickers_list)))
        ax6.set_yticklabels(tickers_list, fontsize=20)
        
        # Add text annotations
        for i in range(len(tickers_list)):
            for j in range(len(criteria_labels)):
                text = '✓' if success_array[i, j] else '✗'
                color = 'white' if success_array[i, j] else 'black'
                ax6.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'executive_summary_dashboard.png', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
        return summary_data
    
    def create_detailed_analysis(self, ticker):
        """Create detailed analysis for a specific ticker"""
        ar_df, beta_df, summary_df = self.load_data(ticker)
        if ar_df is None:
            print(f"No data found for {ticker}")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(30, 24))
        fig.suptitle(f'{ticker} - Detailed Event Study Analysis', fontsize=36, fontweight='bold', y=0.98)
        
        # Colors
        color_news = '#e74c3c'
        color_non_news = '#3498db'
        
        # 1. Abnormal Returns Over Time
        ax1 = axes[0, 0]
        
        # Plot all abnormal returns
        ar_df['AR_pct'] = ar_df['Abnormal_Return'] * 100
        ax1.plot(ar_df.index, ar_df['AR_pct'], color='lightblue', alpha=0.6, linewidth=0.8)
        
        # Highlight news days
        news_days = ar_df[ar_df['News_Day'] == True]
        ax1.scatter(news_days.index, news_days['AR_pct'], 
                   color=color_news, s=60, alpha=0.8, label='Event Days', zorder=5)
        
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Abnormal Returns Over Time', fontweight='bold', fontsize=26)
        ax1.set_ylabel('Abnormal Return (%)', fontsize=22)
        ax1.legend(fontsize=20)
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution Comparison
        ax2 = axes[0, 1]
        
        news_ar = ar_df[ar_df['News_Day'] == True]['AR_pct'].dropna()
        non_news_ar = ar_df[ar_df['News_Day'] == False]['AR_pct'].dropna()
        
        ax2.hist(non_news_ar, bins=30, alpha=0.6, label='Non-Event Days', 
                color=color_non_news, density=True)
        ax2.hist(news_ar, bins=30, alpha=0.8, label='Event Days', 
                color=color_news, density=True)
        
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Abnormal Returns Distribution', fontweight='bold', fontsize=26)
        ax2.set_xlabel('Abnormal Return (%)', fontsize=22)
        ax2.set_ylabel('Density', fontsize=22)
        ax2.legend(fontsize=20)
        
        # 3. Box Plot Comparison
        ax3 = axes[0, 2]
        
        data_for_box = [non_news_ar, news_ar]
        labels = ['Non-Event Days', 'Event Days']
        colors = [color_non_news, color_news]
        
        bp = ax3.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Abnormal Returns Comparison', fontweight='bold', fontsize=26)
        ax3.set_ylabel('Abnormal Return (%)', fontsize=22)
        
        # 4. Model Fit Over Time
        ax4 = axes[1, 0]
        
        if beta_df is not None:
            beta_df['R2_pct'] = beta_df['R_squared'] * 100
            ax4.plot(beta_df.index, beta_df['R2_pct'], color='green', linewidth=2)
            ax4.axhline(y=beta_df['R2_pct'].mean(), color='red', linestyle='--', 
                       alpha=0.7, label=f'Mean: {beta_df["R2_pct"].mean():.1f}%')
            
            ax4.set_title('Model Fit (R²) Over Time', fontweight='bold', fontsize=26)
            ax4.set_ylabel('R² (%)', fontsize=22)
            ax4.legend(fontsize=20)
            ax4.grid(True, alpha=0.3)
        
        # 5. Beta Stability
        ax5 = axes[1, 1]
        
        if beta_df is not None:
            beta_cols = [col for col in beta_df.columns if col.startswith('Beta_')]
            if beta_cols:
                beta_data = beta_df[beta_cols].dropna()
                beta_std = beta_data.std()
                beta_mean = beta_data.mean()
                cv = beta_std / beta_mean.abs()
                
                bars = ax5.bar(range(len(cv)), cv.values, alpha=0.7, color='orange')
                ax5.set_title('Beta Stability (Coefficient of Variation)', fontweight='bold', fontsize=26)
                ax5.set_ylabel('CV (Lower = More Stable)', fontsize=22)
                ax5.set_xticks(range(len(cv)))
                ax5.set_xticklabels([col.replace('Beta_', '') for col in cv.index], rotation=45, fontsize=18)
                
                # Add value labels
                for bar, value in zip(bars, cv.values):
                    height = bar.get_height()
                    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Summary Statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create summary text
        summary_text = f"""
        ANALYSIS SUMMARY
        
        Total Days: {len(ar_df):,}
        Event Days: {len(news_days):,}
        Event Frequency: {len(news_days)/len(ar_df)*100:.1f}%
        
        Mean AR (Events): {news_ar.mean():.3f}%
        Mean AR (Non-Events): {non_news_ar.mean():.3f}%
        Impact Magnitude: {abs(news_ar.mean() - non_news_ar.mean()):.3f}%
        
        Model Quality (R²): {f"{beta_df['R_squared'].mean():.3f}" if beta_df is not None else "N/A"}
        
        Assessment:
        {('✓ Economic Impact' if abs(news_ar.mean() - non_news_ar.mean()) > 0.5 else '✗ Small Impact')}
        {('✓ Good Model Fit' if beta_df['R_squared'].mean() > 0.4 else '✗ Poor Model Fit') if beta_df is not None else '✗ No Data'}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{ticker}_detailed_analysis.png', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_methodology_chart(self):
        """Create methodology explanation chart"""
        fig, ax = plt.subplots(1, 1, figsize=(28, 22))
        ax.set_xlim(0, 18)
        ax.set_ylim(0, 18)
        ax.axis('off')
        
        # Title
        ax.text(9, 17.5, 'Event Study Methodology', fontsize=36, fontweight='bold', ha='center')
        
        # Step 1 - Massive box
        ax.add_patch(plt.Rectangle((1, 14.5), 7, 2, facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=3))
        ax.text(4.5, 15.8, 'Step 1: Estimate Expected Returns', fontsize=22, fontweight='bold', ha='center')
        ax.text(4.5, 15.3, 'Fama-French 5-Factor Model', fontsize=20, ha='center', style='italic')
        ax.text(4.5, 14.9, 'E(R) = α + β₁×MKT + β₂×SMB + β₃×HML + β₄×RMW + β₅×CMA', 
                fontsize=18, ha='center', fontfamily='monospace', weight='bold')
        
        # Step 2 - Massive box
        ax.add_patch(plt.Rectangle((10, 14.5), 7, 2, facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=3))
        ax.text(13.5, 15.8, 'Step 2: Calculate Abnormal Returns', fontsize=22, fontweight='bold', ha='center')
        ax.text(13.5, 15.3, 'Difference from Expected', fontsize=20, ha='center', style='italic')
        ax.text(13.5, 14.9, 'AR = Actual Return - Expected Return', 
                fontsize=18, ha='center', fontfamily='monospace', weight='bold')
        
        # Step 3 - Larger box
        ax.add_patch(plt.Rectangle((0.5, 7.5), 5, 1.2, facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=2))
        ax.text(3, 8.3, 'Step 3: Identify Event Days', fontsize=16, fontweight='bold', ha='center')
        ax.text(3, 7.9, 'Major Corporate Events', fontsize=14, ha='center', style='italic')
        ax.text(3, 7.7, '• Earnings announcements\n• Product launches\n• Executive changes', 
                fontsize=13, ha='center', linespacing=1.2)
        
        # Step 4 - Larger box
        ax.add_patch(plt.Rectangle((6.5, 7.5), 5, 1.2, facecolor='lightcoral', alpha=0.8, edgecolor='black', linewidth=2))
        ax.text(9, 8.3, 'Step 4: Statistical Testing', fontsize=16, fontweight='bold', ha='center')
        ax.text(9, 7.9, 'Significance & Impact', fontsize=14, ha='center', style='italic')
        ax.text(9, 7.7, '• t-tests for significance\n• Effect size (Cohen\'s d)\n• Economic impact assessment', 
                fontsize=13, ha='center', linespacing=1.2)
        
        # Data Flow Arrows - Thicker
        ax.arrow(3, 9.3, 3, 0, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=3)
        ax.arrow(9, 9.3, -3, 0, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=3)
        ax.arrow(3, 7.3, 0, -1.8, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=3)
        ax.arrow(9, 7.3, 0, -1.8, head_width=0.15, head_length=0.2, fc='black', ec='black', linewidth=3)
        
        # Key Assumptions - Larger box
        ax.add_patch(plt.Rectangle((1, 5), 10, 1.8, facecolor='lightgray', alpha=0.7, edgecolor='black', linewidth=2))
        ax.text(6, 6.2, 'Key Assumptions & Parameters', fontsize=18, fontweight='bold', ha='center')
        assumptions_text = [
            '• Rolling window: 126 trading days (~6 months)',
            '• Minimum periods: 50 days',
            '• Event window: Day 0 only (no exclusion window)',
            '• Fama-French 5-Factor Model for expected returns'
        ]
        for i, line in enumerate(assumptions_text):
            ax.text(6, 5.6 - i*0.15, line, fontsize=14, ha='center')
        
        # Success Criteria - Larger box
        ax.add_patch(plt.Rectangle((1, 2.5), 10, 1.8, facecolor='lightpink', alpha=0.7, edgecolor='black', linewidth=2))
        ax.text(6, 3.7, 'Success Criteria', fontsize=18, fontweight='bold', ha='center')
        criteria_text = [
            '• Statistical significance (p < 0.05)',
            '• Economic impact magnitude (>1%)',
            '• Model quality (R² > 0.4)',
            '• Effect size (Cohen\'s d > 0.2)'
        ]
        for i, line in enumerate(criteria_text):
            ax.text(6, 3.1 - i*0.15, line, fontsize=14, ha='center')
        
        # Data Sources - New section
        ax.add_patch(plt.Rectangle((1, 0.5), 10, 1.5, facecolor='lightcyan', alpha=0.7, edgecolor='black', linewidth=2))
        ax.text(6, 1.5, 'Data Sources & Quality', fontsize=18, fontweight='bold', ha='center')
        data_text = [
            '• Stock prices: Daily closing prices',
            '• News events: EODHD filtered dataset',
            '• Market factors: Fama-French 5-Factor data',
            '• Analysis period: 2020-2024'
        ]
        for i, line in enumerate(data_text):
            ax.text(6, 1.0 - i*0.15, line, fontsize=14, ha='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'methodology_explanation.png', 
                   bbox_inches='tight', facecolor='white')
        plt.close()
    
    def create_all_visualizations(self):
        """Create all visualizations for presentation"""
        print("Creating professional visualizations for stakeholder presentation...")
        
        # Create executive summary dashboard
        print("  ✓ Creating executive summary dashboard...")
        summary_data = self.create_overview_dashboard()
        
        # Create detailed analysis for each ticker
        for ticker in ['AAPL', 'TSLA']:
            print(f"  ✓ Creating detailed analysis for {ticker}...")
            self.create_detailed_analysis(ticker)
        
        # Create methodology explanation
        print("  ✓ Creating methodology explanation...")
        self.create_methodology_chart()
        
        print(f"\n🎉 All visualizations saved to: {self.output_dir.absolute()}")
        print("\nFiles created:")
        print("  • executive_summary_dashboard.png")
        print("  • AAPL_detailed_analysis.png")
        print("  • TSLA_detailed_analysis.png")
        print("  • methodology_explanation.png")
        
        return summary_data


def main():
    """Create all presentation visualizations"""
    visualizer = PresentationVisualizer()
    summary_data = visualizer.create_all_visualizations()
    
    # Print summary for reference
    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY FOR STAKEHOLDERS")
    print("="*60)
    
    for data in summary_data:
        print(f"\n{data['Ticker']}:")
        print(f"  Event Frequency: {data['Event Frequency']:.1f}%")
        print(f"  Economic Impact: {data['Impact']:.2f}%")
        print(f"  Assessment: {'Significant' if abs(data['Impact']) > 1.0 else 'Minimal'}")


if __name__ == "__main__":
    main()
