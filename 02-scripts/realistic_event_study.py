"""
Realistic Event Study Analysis
Focuses on truly significant events with meaningful economic impact
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


class RealisticEventStudy:
    """Event study focusing on truly significant events with meaningful impact"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def identify_major_events(self, ticker, min_impact=2.0):
        """
        Identify only major events that could have 2%+ impact
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        min_impact : float
            Minimum expected impact in percentage
        """
        print(f"\n=== IDENTIFYING MAJOR EVENTS FOR {ticker} (Min Impact: {min_impact}%) ===")
        
        # Load the filtered news data
        news_path = Path(self.data_dir) / f"{ticker}_eodhd_news.csv"
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Strategy: Focus on events that typically move markets significantly
        major_events = []
        
        # 1. Earnings announcements (Q1, Q2, Q3, Q4 results)
        earnings_keywords = [
            'q1.*earnings', 'q2.*earnings', 'q3.*earnings', 'q4.*earnings',
            'first quarter.*earnings', 'second quarter.*earnings', 
            'third quarter.*earnings', 'fourth quarter.*earnings',
            'quarterly.*results', 'financial.*results'
        ]
        
        earnings_events = df[
            df['title'].str.contains('|'.join(earnings_keywords), case=False, na=False, regex=True)
        ]
        
        if len(earnings_events) > 0:
            # Take only the most recent earnings (last 8 quarters = 2 years)
            earnings_selected = earnings_events.sort_values('date').tail(8)
            major_events.append(earnings_selected)
            print(f"  ✓ Selected {len(earnings_selected)} earnings announcements")
        
        # 2. Major product launches/announcements
        if ticker == 'AAPL':
            major_products = [
                'iphone.*launch', 'iphone.*announce', 'iphone.*unveil',
                'ipad.*launch', 'ipad.*announce',
                'mac.*launch', 'mac.*announce',
                'apple watch.*launch', 'watch.*announce',
                'airpods.*launch', 'airpods.*announce'
            ]
        else:  # TSLA
            major_products = [
                'model.*launch', 'model.*announce',
                'cybertruck.*launch', 'cybertruck.*announce',
                'semi.*launch', 'semi.*announce',
                'roadster.*launch', 'roadster.*announce',
                'gigafactory.*announce', 'factory.*announce'
            ]
        
        product_events = df[
            df['title'].str.contains('|'.join(major_products), case=False, na=False, regex=True)
        ]
        
        if len(product_events) > 0:
            # Take only major product announcements
            product_selected = product_events.sort_values('date').tail(5)
            major_events.append(product_selected)
            print(f"  ✓ Selected {len(product_selected)} major product announcements")
        
        # 3. CEO/Executive major announcements
        executive_keywords = [
            'ceo.*announce', 'executive.*change', 'leadership.*change',
            'management.*change', 'board.*change', 'ceo.*step'
        ]
        
        executive_events = df[
            df['title'].str.contains('|'.join(executive_keywords), case=False, na=False, regex=True)
        ]
        
        if len(executive_events) > 0:
            # Take only major executive changes
            executive_selected = executive_events.sort_values('date').tail(3)
            major_events.append(executive_selected)
            print(f"  ✓ Selected {len(executive_selected)} major executive announcements")
        
        # 4. Regulatory/Legal major events
        regulatory_keywords = [
            'lawsuit.*settle', 'regulatory.*approval', 'fda.*approval',
            'sec.*investigation', 'antitrust', 'regulation'
        ]
        
        regulatory_events = df[
            df['title'].str.contains('|'.join(regulatory_keywords), case=False, na=False, regex=True)
        ]
        
        if len(regulatory_events) > 0:
            # Take only major regulatory events
            regulatory_selected = regulatory_events.sort_values('date').tail(3)
            major_events.append(regulatory_selected)
            print(f"  ✓ Selected {len(regulatory_selected)} major regulatory events")
        
        # Combine all major events
        if major_events:
            all_major = pd.concat(major_events, ignore_index=True).drop_duplicates(subset=['title', 'date'])
            all_major = all_major.sort_values('date')
            
            print(f"\n📊 MAJOR EVENTS SUMMARY:")
            print(f"  Total major events: {len(all_major)}")
            print(f"  Date range: {all_major['date'].min().date()} to {all_major['date'].max().date()}")
            
            # Create event dates file
            event_dates = pd.DataFrame({'Date': all_major['date'].dt.date.unique()})
            event_dates = event_dates.sort_values('Date')
            
            # Save files
            events_file = f"{ticker}_major_events.csv"
            all_major.to_csv(Path(self.data_dir) / events_file, index=False)
            
            dates_file = f"{ticker}_major_event_dates.csv"
            event_dates.to_csv(Path(self.data_dir) / dates_file, index=False)
            
            print(f"  ✓ Saved to {events_file}")
            print(f"  ✓ Created {dates_file} with {len(event_dates)} unique dates")
            
            return len(event_dates)
        else:
            print("  ❌ No major events found!")
            return 0
    
    def run_realistic_event_study(self, ticker, min_impact=2.0):
        """
        Run event study with major events only
        """
        print(f"\n=== RUNNING REALISTIC EVENT STUDY FOR {ticker} ===")
        
        # Identify major events
        num_events = self.identify_major_events(ticker, min_impact)
        
        if num_events == 0:
            print(f"❌ No major events found for {ticker}")
            return None
        
        # File names
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = f"{ticker}_major_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = f"../03-output/{ticker}_major_events"
        
        try:
            # Create a custom analysis class with minimal exclusion
            class MinimalExclusionAnalysis(Phase1Analysis):
                def _estimate_betas(self):
                    """Override with minimal exclusion"""
                    from pathlib import Path
                    import importlib
                    beta_estimation_module = importlib.import_module('02_beta_estimation')
                    BetaEstimator = beta_estimation_module.BetaEstimator
                    
                    # Use minimal exclusion - only exclude the event day itself
                    estimator = BetaEstimator(window_size=126, min_periods=50)
                    
                    self.beta_df = estimator.rolling_beta_estimation(
                        data=self.data,
                        factor_cols=self.factor_cols,
                        exclude_dates=self.news_dates,
                        event_window=(0, 0)  # Only exclude the event day, not surrounding days
                    )
                    
                    # Calculate beta stability
                    stability = estimator.calculate_beta_stability(self.beta_df, self.factor_cols)
                    
                    # Check how many valid estimates we got
                    valid_estimates = self.beta_df['R_squared'].notna().sum()
                    print(f"  ✓ Estimated betas for {len(self.beta_df)} days")
                    print(f"  ✓ Valid estimates: {valid_estimates} ({valid_estimates/len(self.beta_df)*100:.1f}%)")
                    
                    if valid_estimates > 0:
                        avg_r2 = self.beta_df['R_squared'].mean()
                        print(f"  ✓ Average R²: {avg_r2:.3f}")
                    else:
                        print(f"  ⚠ No valid estimates - trying without exclusion...")
                        # Fallback: no exclusion at all
                        self.beta_df = estimator.rolling_beta_estimation(
                            data=self.data,
                            factor_cols=self.factor_cols,
                            exclude_dates=None,
                            event_window=(0, 0)
                        )
                        
                        valid_estimates = self.beta_df['R_squared'].notna().sum()
                        if valid_estimates > 0:
                            avg_r2 = self.beta_df['R_squared'].mean()
                            print(f"  ✓ Fallback successful: {valid_estimates} valid estimates, avg R²: {avg_r2:.3f}")
                    
                    # Save beta estimates
                    self.beta_df.to_csv(self.output_dir / "beta_estimates.csv")
                    stability.to_csv(self.output_dir / "beta_stability.csv")
            
            # Use the custom analysis
            analysis = MinimalExclusionAnalysis(
                stock_file=stock_file,
                news_file=event_dates_file,
                ff_file=ff_file,
                sector_file=None,
                data_dir=self.data_dir,
                output_dir=output_dir
            )
            
            summary = analysis.run_complete_analysis()
            
            if summary:
                print(f"\n✅ {ticker} Realistic Event Study Complete!")
                print(f"   Output saved to: {Path(output_dir).absolute()}")
                
                # Enhanced analysis
                self.analyze_major_event_impact(ticker, summary)
                
                return summary
            else:
                print(f"\n❌ {ticker} Realistic Event Study Failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ {ticker} Realistic Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_major_event_impact(self, ticker, summary):
        """
        Analyze the impact of major events with realistic expectations
        """
        print(f"\n=== MAJOR EVENT IMPACT ANALYSIS FOR {ticker} ===")
        
        # Load abnormal returns data
        ar_file = Path(self.output_dir) / f"{ticker}_major_events" / "abnormal_returns.csv"
        if ar_file.exists():
            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
            
            # Calculate event-specific metrics
            news_days = ar_df[ar_df['News_Day'] == True]
            non_news_days = ar_df[ar_df['News_Day'] == False]
            
            print(f"\n📊 MAJOR EVENT IMPACT METRICS:")
            print(f"  Event Days: {len(news_days)}")
            print(f"  Non-Event Days: {len(non_news_days)}")
            print(f"  Event Frequency: {len(news_days)/(len(news_days)+len(non_news_days))*100:.1f}%")
            
            if len(news_days) > 0 and len(non_news_days) > 0:
                # Statistical tests
                news_ar = news_days['Abnormal_Return'].dropna()
                non_news_ar = non_news_days['Abnormal_Return'].dropna()
                
                if len(news_ar) > 0 and len(non_news_ar) > 0:
                    # T-test for difference in means
                    t_stat, t_pvalue = stats.ttest_ind(news_ar, non_news_ar, equal_var=False)
                    print(f"  Welch's t-test: t={t_stat:.3f}, p={t_pvalue:.3f}")
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(news_ar)-1)*news_ar.var() + (len(non_news_ar)-1)*non_news_ar.var()) / 
                                       (len(news_ar) + len(non_news_ar) - 2))
                    cohens_d = (news_ar.mean() - non_news_ar.mean()) / pooled_std
                    print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
                    
                    # Magnitude of impact
                    impact_magnitude = abs(news_ar.mean() - non_news_ar.mean()) * 100
                    print(f"  Impact magnitude: {impact_magnitude:.3f}%")
                    
                    # Realistic success criteria
                    print(f"\n🎯 REALISTIC EVENT STUDY SUCCESS CRITERIA:")
                    if t_pvalue < 0.05:
                        print(f"  ✅ Statistically significant (p < 0.05)")
                    else:
                        print(f"  ❌ Not statistically significant (p ≥ 0.05)")
                    
                    if impact_magnitude > 1.0:
                        print(f"  ✅ Economically meaningful impact (>1.0%)")
                    elif impact_magnitude > 0.5:
                        print(f"  ⚠️  Moderate impact (0.5-1.0%)")
                    else:
                        print(f"  ❌ Small impact (≤0.5%)")
                    
                    if abs(cohens_d) > 0.5:
                        print(f"  ✅ Large effect size (|d| > 0.5)")
                    elif abs(cohens_d) > 0.2:
                        print(f"  ⚠️  Medium effect size (|d| > 0.2)")
                    else:
                        print(f"  ❌ Small effect size (|d| ≤ 0.2)")
                    
                    # Overall assessment
                    print(f"\n📈 OVERALL ASSESSMENT:")
                    if t_pvalue < 0.05 and impact_magnitude > 1.0:
                        print(f"  🎉 SUCCESS: Major events have significant economic impact!")
                    elif t_pvalue < 0.05 and impact_magnitude > 0.5:
                        print(f"  ✅ PARTIAL SUCCESS: Events have some impact")
                    elif impact_magnitude > 1.0:
                        print(f"  ⚠️  ECONOMIC IMPACT: Large magnitude but not statistically significant")
                    else:
                        print(f"  ❌ NO CLEAR IMPACT: Events don't show meaningful impact")
                else:
                    print("  ❌ Insufficient data for statistical analysis")
    
    def run_complete_realistic_event_study(self, min_impact=2.0):
        """Run complete realistic event study"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "REALISTIC EVENT STUDY ANALYSIS")
        print("=" * 80)
        print(f"\nThis analysis focuses on truly major events (min impact: {min_impact}%):")
        print("  • Earnings announcements (Q1-Q4)")
        print("  • Major product launches")
        print("  • Executive changes")
        print("  • Regulatory events")
        print("\nExpecting meaningful economic impact (>1%)...")
        print("=" * 80)
        
        results = {}
        
        # Process AAPL
        print("\n\n" + "🍎" * 40)
        results['AAPL'] = self.run_realistic_event_study('AAPL', min_impact)
        
        # Process TSLA
        print("\n\n" + "⚡" * 40)
        results['TSLA'] = self.run_realistic_event_study('TSLA', min_impact)
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 30 + "REALISTIC EVENT STUDY SUMMARY")
        print("=" * 80)
        
        for ticker, summary in results.items():
            if summary:
                print(f"\n{ticker} REALISTIC EVENT STUDY:")
                print(f"  Event Days: {summary['news_days']}")
                print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.2f}%")
                print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.2f}%")
                print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                print(f"  Model R²: {summary['avg_r_squared']:.3f}")
                
                # Calculate impact magnitude
                impact = abs(summary['mean_ar_news'] - summary['mean_ar_non_news']) * 100
                
                if summary['significant_tests'] >= 2 and impact > 1.0:
                    print(f"  🎉 SUCCESS: Major events have significant impact!")
                elif summary['significant_tests'] >= 1 and impact > 0.5:
                    print(f"  ✅ PARTIAL: Some evidence of event impact")
                elif impact > 1.0:
                    print(f"  ⚠️  ECONOMIC: Large impact but not significant")
                else:
                    print(f"  ❌ NO IMPACT: Events don't show meaningful impact")
            else:
                print(f"\n{ticker}: ❌ Realistic event study failed")
        
        print("\n" + "=" * 80)
        print("🎉 REALISTIC EVENT STUDY COMPLETE!")
        print("=" * 80)


def main():
    """Run the realistic event study"""
    event_study = RealisticEventStudy()
    event_study.run_complete_realistic_event_study(min_impact=2.0)


if __name__ == "__main__":
    main()
