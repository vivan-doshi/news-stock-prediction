"""
Improved Event Study Analysis
Focuses on significant events with proper methodology for high-frequency news
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


class ImprovedEventStudy:
    """Enhanced event study with better methodology for high-frequency news"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def identify_significant_events(self, news_file, ticker):
        """
        Identify only the most significant news events
        
        Parameters:
        -----------
        news_file : str
            Path to filtered news file
        ticker : str
            Stock ticker
        """
        print(f"\n=== IDENTIFYING SIGNIFICANT EVENTS FOR {ticker} ===")
        
        # Load filtered news data
        news_path = Path(self.data_dir) / news_file
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Define significant event criteria
        significant_events = []
        
        # 1. Earnings announcements (highest impact)
        earnings_keywords = ['earnings', 'quarterly results', 'financial results', 'q1', 'q2', 'q3', 'q4']
        earnings_events = df[
            df['title'].str.contains('|'.join(earnings_keywords), case=False, na=False) |
            df['content'].str.contains('|'.join(earnings_keywords), case=False, na=False)
        ]
        significant_events.append(earnings_events)
        print(f"  ✓ Found {len(earnings_events)} earnings-related events")
        
        # 2. Product launches (major announcements)
        product_keywords = ['launch', 'release', 'unveil', 'introduce', 'new product', 'new model']
        product_events = df[
            df['title'].str.contains('|'.join(product_keywords), case=False, na=False) |
            df['content'].str.contains('|'.join(product_keywords), case=False, na=False)
        ]
        significant_events.append(product_events)
        print(f"  ✓ Found {len(product_events)} product launch events")
        
        # 3. Strong sentiment events (very positive or negative)
        strong_sentiment = df[
            (df['sentiment_polarity'] > 0.7) | (df['sentiment_polarity'] < -0.7)
        ]
        significant_events.append(strong_sentiment)
        print(f"  ✓ Found {len(strong_sentiment)} strong sentiment events")
        
        # 4. CEO/executive announcements
        executive_keywords = ['ceo', 'executive', 'leadership', 'management', 'board']
        executive_events = df[
            df['title'].str.contains('|'.join(executive_keywords), case=False, na=False) |
            df['content'].str.contains('|'.join(executive_keywords), case=False, na=False)
        ]
        significant_events.append(executive_events)
        print(f"  ✓ Found {len(executive_events)} executive-related events")
        
        # Combine and deduplicate
        all_significant = pd.concat(significant_events, ignore_index=True).drop_duplicates(subset=['title', 'date'])
        
        # Sort by date
        all_significant = all_significant.sort_values('date')
        
        print(f"\n📊 SIGNIFICANT EVENTS SUMMARY:")
        print(f"  Total significant events: {len(all_significant)}")
        print(f"  Date range: {all_significant['date'].min().date()} to {all_significant['date'].max().date()}")
        
        # Create event dates file
        event_dates = pd.DataFrame({'Date': all_significant['date'].dt.date.unique()})
        event_dates = event_dates.sort_values('Date')
        
        # Save significant events
        events_file = f"{ticker}_significant_events.csv"
        all_significant.to_csv(Path(self.data_dir) / events_file, index=False)
        
        # Save event dates
        dates_file = f"{ticker}_event_dates.csv"
        event_dates.to_csv(Path(self.data_dir) / dates_file, index=False)
        
        print(f"  ✓ Saved {len(all_significant)} events to {events_file}")
        print(f"  ✓ Created {dates_file} with {len(event_dates)} unique event dates")
        
        return len(event_dates)
    
    def run_event_study(self, ticker, event_window=(-2, 2)):
        """
        Run proper event study with significant events only
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        event_window : tuple
            Event window (days before, days after)
        """
        print(f"\n=== RUNNING EVENT STUDY FOR {ticker} ===")
        
        # File names
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = f"{ticker}_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = f"../03-output/{ticker}_event_study"
        
        try:
            # Use the standard analysis with significant events only
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
                print(f"\n✅ {ticker} Event Study Complete!")
                print(f"   Output saved to: {Path(output_dir).absolute()}")
                
                # Enhanced analysis
                self.analyze_event_impact(ticker, summary, event_window)
                
                return summary
            else:
                print(f"\n❌ {ticker} Event Study Failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ {ticker} Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_event_impact(self, ticker, summary, event_window):
        """
        Analyze the impact of events with enhanced metrics
        """
        print(f"\n=== ENHANCED EVENT IMPACT ANALYSIS FOR {ticker} ===")
        
        # Load abnormal returns data
        ar_file = Path(self.output_dir) / f"{ticker}_event_study" / "abnormal_returns.csv"
        if ar_file.exists():
            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
            
            # Calculate event-specific metrics
            news_days = ar_df[ar_df['News_Day'] == True]
            non_news_days = ar_df[ar_df['News_Day'] == False]
            
            print(f"\n📊 EVENT IMPACT METRICS:")
            print(f"  News Days: {len(news_days)}")
            print(f"  Non-News Days: {len(non_news_days)}")
            
            if len(news_days) > 0 and len(non_news_days) > 0:
                # Statistical tests
                news_ar = news_days['Abnormal_Return'].dropna()
                non_news_ar = non_news_days['Abnormal_Return'].dropna()
                
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
                print(f"\n🎯 EVENT STUDY SUCCESS CRITERIA:")
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
    
    def run_complete_event_study(self):
        """Run complete event study for both tickers"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "IMPROVED EVENT STUDY ANALYSIS")
        print("=" * 80)
        print("\nThis analysis focuses on significant events only:")
        print("  • Earnings announcements")
        print("  • Product launches")
        print("  • Strong sentiment events")
        print("  • Executive announcements")
        print("\nUsing proper event study methodology...")
        print("=" * 80)
        
        results = {}
        
        # Process AAPL
        print("\n\n" + "🍎" * 40)
        aapl_events = self.identify_significant_events("AAPL_eodhd_news.csv", "AAPL")
        if aapl_events > 0:
            results['AAPL'] = self.run_event_study('AAPL')
        else:
            print("❌ No significant events found for AAPL")
            results['AAPL'] = None
        
        # Process TSLA
        print("\n\n" + "⚡" * 40)
        tsla_events = self.identify_significant_events("TSLA_eodhd_news.csv", "TSLA")
        if tsla_events > 0:
            results['TSLA'] = self.run_event_study('TSLA')
        else:
            print("❌ No significant events found for TSLA")
            results['TSLA'] = None
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 30 + "EVENT STUDY SUMMARY")
        print("=" * 80)
        
        for ticker, summary in results.items():
            if summary:
                print(f"\n{ticker} EVENT STUDY:")
                print(f"  Event Days: {summary['news_days']}")
                print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.2f}%")
                print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.2f}%")
                print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                print(f"  Model R²: {summary['avg_r_squared']:.3f}")
                
                if summary['significant_tests'] >= 2:
                    print(f"  ✅ SUCCESS: Significant event impact detected!")
                elif summary['significant_tests'] >= 1:
                    print(f"  ⚠️  PARTIAL: Some evidence of event impact")
                else:
                    print(f"  ❌ FAILURE: No significant event impact")
            else:
                print(f"\n{ticker}: ❌ Event study failed")
        
        print("\n" + "=" * 80)
        print("🎉 IMPROVED EVENT STUDY COMPLETE!")
        print("=" * 80)
        print("\n📁 Results saved to:")
        print("   - 03-output/AAPL_event_study/")
        print("   - 03-output/TSLA_event_study/")
        print("=" * 80)


def main():
    """Run the improved event study"""
    event_study = ImprovedEventStudy()
    event_study.run_complete_event_study()


if __name__ == "__main__":
    main()
