"""
Simple Event Study Analysis
Focuses on truly rare events with minimal exclusion
"""

import sys
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
main_analysis_module = importlib.import_module('05_main_analysis')
Phase1Analysis = main_analysis_module.Phase1Analysis


class SimpleEventStudy:
    """Simple event study focusing on rare events with minimal exclusion"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def create_rare_events(self, ticker, max_events=30):
        """
        Create a small set of rare events manually
        """
        print(f"\n=== CREATING RARE EVENTS FOR {ticker} ===")
        
        # Load the filtered news data
        news_path = Path(self.data_dir) / f"{ticker}_eodhd_news.csv"
        df = pd.read_csv(news_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Strategy: Select events with very specific criteria
        rare_events = []
        
        # 1. Earnings announcements (most important)
        earnings_patterns = [
            'earnings report',
            'quarterly earnings', 
            'q1 earnings',
            'q2 earnings', 
            'q3 earnings',
            'q4 earnings',
            'financial results'
        ]
        
        earnings_events = df[
            df['title'].str.contains('|'.join(earnings_patterns), case=False, na=False)
        ]
        
        if len(earnings_events) > 0:
            # Take only the most recent earnings (last 12 quarters = 3 years)
            earnings_selected = earnings_events.sort_values('date').tail(12)
            rare_events.append(earnings_selected)
            print(f"  ✓ Selected {len(earnings_selected)} earnings events")
        
        # 2. Major product launches (very specific)
        if ticker == 'AAPL':
            product_patterns = [
                'iphone.*launch',
                'ipad.*launch', 
                'mac.*launch',
                'apple watch.*launch',
                'airpods.*launch'
            ]
        else:  # TSLA
            product_patterns = [
                'model.*launch',
                'cybertruck.*launch',
                'semi.*launch',
                'roadster.*launch',
                'gigafactory'
            ]
        
        product_events = df[
            df['title'].str.contains('|'.join(product_patterns), case=False, na=False, regex=True)
        ]
        
        if len(product_events) > 0:
            # Take only major launches
            product_selected = product_events.sort_values('date').tail(10)
            rare_events.append(product_selected)
            print(f"  ✓ Selected {len(product_selected)} product launch events")
        
        # 3. Strong sentiment events (most extreme)
        strong_sentiment = df[
            (df['sentiment_polarity'] > 0.9) | (df['sentiment_polarity'] < -0.9)
        ]
        
        if len(strong_sentiment) > 0:
            # Take the most extreme ones
            top_positive = strong_sentiment.nlargest(5, 'sentiment_polarity')
            top_negative = strong_sentiment.nsmallest(5, 'sentiment_polarity')
            sentiment_selected = pd.concat([top_positive, top_negative]).drop_duplicates()
            rare_events.append(sentiment_selected)
            print(f"  ✓ Selected {len(sentiment_selected)} extreme sentiment events")
        
        # Combine all rare events
        if rare_events:
            all_rare = pd.concat(rare_events, ignore_index=True).drop_duplicates(subset=['title', 'date'])
            all_rare = all_rare.sort_values('date')
            
            # Limit to max_events
            if len(all_rare) > max_events:
                all_rare = all_rare.tail(max_events)
            
            print(f"\n📊 RARE EVENTS SUMMARY:")
            print(f"  Total rare events: {len(all_rare)}")
            print(f"  Date range: {all_rare['date'].min().date()} to {all_rare['date'].max().date()}")
            
            # Create event dates file
            event_dates = pd.DataFrame({'Date': all_rare['date'].dt.date.unique()})
            event_dates = event_dates.sort_values('Date')
            
            # Save files
            events_file = f"{ticker}_rare_events.csv"
            all_rare.to_csv(Path(self.data_dir) / events_file, index=False)
            
            dates_file = f"{ticker}_rare_event_dates.csv"
            event_dates.to_csv(Path(self.data_dir) / dates_file, index=False)
            
            print(f"  ✓ Saved to {events_file}")
            print(f"  ✓ Created {dates_file} with {len(event_dates)} unique dates")
            
            return len(event_dates)
        else:
            print("  ❌ No rare events found!")
            return 0
    
    def run_simple_event_study(self, ticker, max_events=30):
        """
        Run event study with rare events
        """
        print(f"\n=== RUNNING SIMPLE EVENT STUDY FOR {ticker} ===")
        
        # Create rare events
        num_events = self.create_rare_events(ticker, max_events)
        
        if num_events == 0:
            print(f"❌ No rare events found for {ticker}")
            return None
        
        # File names
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = f"{ticker}_rare_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = f"../03-output/{ticker}_simple_events"
        
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
                print(f"\n✅ {ticker} Simple Event Study Complete!")
                print(f"   Output saved to: {Path(output_dir).absolute()}")
                return summary
            else:
                print(f"\n❌ {ticker} Simple Event Study Failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ {ticker} Simple Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_simple_event_study(self, max_events=30):
        """Run complete simple event study"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "SIMPLE EVENT STUDY ANALYSIS")
        print("=" * 80)
        print(f"\nThis analysis focuses on truly rare events (max {max_events} events):")
        print("  • Earnings announcements (last 12 quarters)")
        print("  • Major product launches")
        print("  • Extreme sentiment events")
        print("\nUsing minimal exclusion to avoid NaN issues...")
        print("=" * 80)
        
        results = {}
        
        # Process AAPL
        print("\n\n" + "🍎" * 40)
        results['AAPL'] = self.run_simple_event_study('AAPL', max_events)
        
        # Process TSLA
        print("\n\n" + "⚡" * 40)
        results['TSLA'] = self.run_simple_event_study('TSLA', max_events)
        
        # Final summary
        print("\n" + "=" * 80)
        print(" " * 30 + "SIMPLE EVENT STUDY SUMMARY")
        print("=" * 80)
        
        for ticker, summary in results.items():
            if summary:
                print(f"\n{ticker} SIMPLE EVENT STUDY:")
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
                print(f"\n{ticker}: ❌ Simple event study failed")
        
        print("\n" + "=" * 80)
        print("🎉 SIMPLE EVENT STUDY COMPLETE!")
        print("=" * 80)


def main():
    """Run the simple event study"""
    event_study = SimpleEventStudy()
    event_study.run_complete_simple_event_study(max_events=30)


if __name__ == "__main__":
    main()
