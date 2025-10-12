"""
Rare Event Study Analysis
Focuses on truly rare, high-impact events only
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


class RareEventStudy:
    """Event study focusing on truly rare, high-impact events"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def identify_rare_events(self, news_file, ticker, max_events=50):
        """
        Identify only the rarest, most significant events
        
        Parameters:
        -----------
        news_file : str
            Path to filtered news file
        ticker : str
            Stock ticker
        max_events : int
            Maximum number of events to select
        """
        print(f"\n=== IDENTIFYING RARE EVENTS FOR {ticker} ===")
        
        # Load filtered news data
        news_path = Path(self.data_dir) / news_file
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Define rare event criteria with high specificity
        rare_events = []
        
        # 1. Earnings announcements only (most specific)
        earnings_keywords = ['earnings report', 'quarterly earnings', 'q[1-4] earnings', 'financial results']
        earnings_events = df[
            df['title'].str.contains('|'.join(earnings_keywords), case=False, na=False, regex=True)
        ]
        
        # Keep only the most recent earnings (typically 4 per year)
        if len(earnings_events) > 0:
            # Sort by date and take the most recent ones
            earnings_events = earnings_events.sort_values('date').tail(16)  # ~4 years * 4 quarters
            rare_events.append(earnings_events)
            print(f"  ✓ Found {len(earnings_events)} earnings announcements")
        
        # 2. Major product launches only (very specific)
        major_launch_keywords = ['unveils', 'announces', 'launches', 'introduces']
        product_events = df[
            df['title'].str.contains('|'.join(major_launch_keywords), case=False, na=False) &
            (df['title'].str.contains('iphone|ipad|mac|watch|airpods', case=False, na=False) if ticker == 'AAPL' else
             df['title'].str.contains('model|cybertruck|semi|roadster', case=False, na=False))
        ]
        
        # Limit to major launches only
        if len(product_events) > 0:
            product_events = product_events.sort_values('date').tail(20)  # Limit to recent major launches
            rare_events.append(product_events)
            print(f"  ✓ Found {len(product_events)} major product launches")
        
        # 3. CEO/executive major announcements only
        executive_keywords = ['ceo announces', 'executive', 'leadership change', 'management']
        executive_events = df[
            df['title'].str.contains('|'.join(executive_keywords), case=False, na=False)
        ]
        
        if len(executive_events) > 0:
            executive_events = executive_events.sort_values('date').tail(10)  # Limit executive events
            rare_events.append(executive_events)
            print(f"  ✓ Found {len(executive_events)} executive announcements")
        
        # 4. Strong sentiment + long content (quality filter)
        quality_events = df[
            ((df['sentiment_polarity'] > 0.8) | (df['sentiment_polarity'] < -0.8)) &
            (df['content'].str.len() > 500)  # Substantial content
        ]
        
        if len(quality_events) > 0:
            # Take the most extreme sentiment events
            quality_events = quality_events.nlargest(20, 'sentiment_polarity').append(
                quality_events.nsmallest(20, 'sentiment_polarity')
            ).drop_duplicates()
            rare_events.append(quality_events)
            print(f"  ✓ Found {len(quality_events)} high-quality sentiment events")
        
        # Combine and deduplicate
        if rare_events:
            all_rare = pd.concat(rare_events, ignore_index=True).drop_duplicates(subset=['title', 'date'])
        else:
            all_rare = pd.DataFrame()
        
        # Sort by date and limit to max_events
        all_rare = all_rare.sort_values('date').tail(max_events)
        
        print(f"\n📊 RARE EVENTS SUMMARY:")
        print(f"  Total rare events: {len(all_rare)}")
        if len(all_rare) > 0:
            print(f"  Date range: {all_rare['date'].min().date()} to {all_rare['date'].max().date()}")
            
            # Create event dates file
            event_dates = pd.DataFrame({'Date': all_rare['date'].dt.date.unique()})
            event_dates = event_dates.sort_values('Date')
            
            # Save rare events
            events_file = f"{ticker}_rare_events.csv"
            all_rare.to_csv(Path(self.data_dir) / events_file, index=False)
            
            # Save event dates
            dates_file = f"{ticker}_rare_event_dates.csv"
            event_dates.to_csv(Path(self.data_dir) / dates_file, index=False)
            
            print(f"  ✓ Saved {len(all_rare)} events to {events_file}")
            print(f"  ✓ Created {dates_file} with {len(event_dates)} unique event dates")
            
            return len(event_dates)
        else:
            print("  ❌ No rare events found!")
            return 0
    
    def run_rare_event_study(self, ticker, max_events=50):
        """
        Run event study with rare events only
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        max_events : int
            Maximum number of events
        """
        print(f"\n=== RUNNING RARE EVENT STUDY FOR {ticker} ===")
        
        # Identify rare events
        num_events = self.identify_rare_events(f"{ticker}_eodhd_news.csv", ticker, max_events)
        
        if num_events == 0:
            print(f"❌ No rare events found for {ticker}")
            return None
        
        # File names
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = f"{ticker}_rare_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = f"../03-output/{ticker}_rare_events"
        
        try:
            # Use the standard analysis with rare events only
            analysis = Phase1Analysis(
                stock_file=stock_file,
                news_file=event_dates_file,
                ff_file=ff_file,
                sector_file=None,
                data_dir=self.data_dir,
                output_dir=output_dir
            )
            
            summary = analysis.run_complete_analysis()
            
            if summary:
                print(f"\n✅ {ticker} Rare Event Study Complete!")
                print(f"   Output saved to: {Path(output_dir).absolute()}")
                
                # Enhanced analysis
                self.analyze_rare_event_impact(ticker, summary)
                
                return summary
            else:
                print(f"\n❌ {ticker} Rare Event Study Failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ {ticker} Rare Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_rare_event_impact(self, ticker, summary):
        """
        Analyze the impact of rare events
        """
        print(f"\n=== RARE EVENT IMPACT ANALYSIS FOR {ticker} ===")
        
        # Load abnormal returns data
        ar_file = Path(self.output_dir) / f"{ticker}_rare_events" / "abnormal_returns.csv"
        if ar_file.exists():
            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
            
            # Calculate event-specific metrics
            news_days = ar_df[ar_df['News_Day'] == True]
            non_news_days = ar_df[ar_df['News_Day'] == False]
            
            print(f"\n📊 RARE EVENT IMPACT METRICS:")
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
                    
                    # Success criteria
                    print(f"\n🎯 RARE EVENT STUDY SUCCESS CRITERIA:")
                    if t_pvalue < 0.05:
                        print(f"  ✅ Statistically significant (p < 0.05)")
                    else:
                        print(f"  ❌ Not statistically significant (p ≥ 0.05)")
                    
                    if abs(cohens_d) > 0.2:
                        print(f"  ✅ Meaningful effect size (|d| > 0.2)")
                    else:
                        print(f"  ❌ Small effect size (|d| ≤ 0.2)")
                    
                    if impact_magnitude > 0.5:
                        print(f"  ✅ Economically significant impact (>0.5%)")
                    else:
                        print(f"  ❌ Small economic impact (≤0.5%)")
                else:
                    print("  ❌ Insufficient data for statistical analysis")
    
    def run_complete_rare_event_study(self, max_events=50):
        """Run complete rare event study for both tickers"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "RARE EVENT STUDY ANALYSIS")
        print("=" * 80)
        print(f"\nThis analysis focuses on truly rare events only (max {max_events} events):")
        print("  • Earnings announcements (4 per year)")
        print("  • Major product launches")
        print("  • Executive announcements")
        print("  • High-quality sentiment events")
        print("\nUsing minimal exclusion to avoid NaN issues...")
        print("=" * 80)
        
        results = {}
        
        # Process AAPL
        print("\n\n" + "🍎" * 40)
        results['AAPL'] = self.run_rare_event_study('AAPL', max_events)
        
        # Process TSLA
        print("\n\n" + "⚡" * 40)
        results['TSLA'] = self.run_rare_event_study('TSLA', max_events)
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 30 + "RARE EVENT STUDY SUMMARY")
        print("=" * 80)
        
        for ticker, summary in results.items():
            if summary:
                print(f"\n{ticker} RARE EVENT STUDY:")
                print(f"  Event Days: {summary['news_days']}")
                print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.2f}%")
                print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.2f}%")
                print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                print(f"  Model R²: {summary['avg_r_squared']:.3f}")
                
                if summary['significant_tests'] >= 2:
                    print(f"  ✅ SUCCESS: Significant rare event impact detected!")
                elif summary['significant_tests'] >= 1:
                    print(f"  ⚠️  PARTIAL: Some evidence of rare event impact")
                else:
                    print(f"  ❌ FAILURE: No significant rare event impact")
            else:
                print(f"\n{ticker}: ❌ Rare event study failed")
        
        print("\n" + "=" * 80)
        print("🎉 RARE EVENT STUDY COMPLETE!")
        print("=" * 80)
        print("\n📁 Results saved to:")
        print("   - 03-output/AAPL_rare_events/")
        print("   - 03-output/TSLA_rare_events/")
        print("=" * 80)


def main():
    """Run the rare event study"""
    event_study = RareEventStudy()
    event_study.run_complete_rare_event_study(max_events=50)


if __name__ == "__main__":
    main()
