"""
Run Event Study with Improved Event Selection
Uses the new improved event selection for better analysis
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


class ImprovedEventStudy:
    """Event study with improved event selection"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def run_improved_event_study(self, ticker):
        """Run event study with improved events"""
        print(f"\n{'='*60}")
        print(f"IMPROVED EVENT STUDY FOR {ticker}")
        print(f"{'='*60}")
        
        # File names for improved events
        stock_file = f"{ticker}_stock_data.csv"
        event_dates_file = f"{ticker}_improved_event_dates.csv"
        ff_file = "fama_french_factors.csv"
        output_dir = f"../03-output/{ticker}_improved_study"
        
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
                print(f"\n✅ {ticker} Improved Event Study Complete!")
                print(f"   Output saved to: {Path(output_dir).absolute()}")
                
                # Enhanced analysis
                self.analyze_improved_results(ticker, summary)
                
                return summary
            else:
                print(f"\n❌ {ticker} Improved Event Study Failed!")
                return None
                
        except Exception as e:
            print(f"\n❌ {ticker} Improved Event Study Failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_improved_results(self, ticker, summary):
        """Analyze the improved event study results"""
        print(f"\n=== IMPROVED RESULTS ANALYSIS FOR {ticker} ===")
        
        # Load abnormal returns data
        ar_file = Path(self.output_dir) / f"{ticker}_improved_study" / "abnormal_returns.csv"
        if ar_file.exists():
            ar_df = pd.read_csv(ar_file, index_col=0, parse_dates=True)
            
            # Calculate event-specific metrics
            news_days = ar_df[ar_df['News_Day'] == True]
            non_news_days = ar_df[ar_df['News_Day'] == False]
            
            print(f"\n📊 IMPROVED EVENT IMPACT METRICS:")
            print(f"  Event Days: {len(news_days)}")
            print(f"  Non-Event Days: {len(non_news_days)}")
            print(f"  Event Frequency: {len(news_days)/(len(news_days)+len(non_news_days))*100:.1f}%")
            
            if len(news_days) > 0 and len(non_news_days) > 0:
                # Statistical tests
                news_ar = news_days['Abnormal_Return'].dropna()
                non_news_ar = non_news_days['Abnormal_Return'].dropna()
                
                if len(news_ar) > 0 and len(non_news_ar) > 0:
                    from scipy import stats
                    
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
                    print(f"\n🎯 IMPROVED EVENT STUDY SUCCESS CRITERIA:")
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
                        print(f"  🎉 SUCCESS: High-impact events show significant economic impact!")
                    elif t_pvalue < 0.05 and impact_magnitude > 0.5:
                        print(f"  ✅ PARTIAL SUCCESS: Events show some statistical impact")
                    elif impact_magnitude > 1.0:
                        print(f"  ⚠️  ECONOMIC IMPACT: Large magnitude but not statistically significant")
                    else:
                        print(f"  ❌ NO CLEAR IMPACT: Events don't show meaningful impact")
                else:
                    print("  ❌ Insufficient data for statistical analysis")
    
    def compare_with_previous(self, ticker):
        """Compare improved results with previous analysis"""
        print(f"\n=== COMPARISON WITH PREVIOUS ANALYSIS FOR {ticker} ===")
        
        # Load improved results
        improved_file = Path(self.output_dir) / f"{ticker}_improved_study" / "analysis_summary.csv"
        if improved_file.exists():
            improved_df = pd.read_csv(improved_file)
            improved_summary = improved_df.iloc[0]
            
            # Load previous results
            previous_file = Path(self.output_dir) / f"{ticker}_major_events" / "analysis_summary.csv"
            if previous_file.exists():
                previous_df = pd.read_csv(previous_file)
                previous_summary = previous_df.iloc[0]
                
                print(f"\n📊 COMPARISON SUMMARY:")
                print(f"  Metric                    Previous    Improved    Change")
                print(f"  {'-'*60}")
                print(f"  Event Days               {previous_summary['news_days']:>8}    {improved_summary['news_days']:>8}    {improved_summary['news_days'] - previous_summary['news_days']:>+8}")
                print(f"  Mean AR (Events)         {previous_summary['mean_ar_news']*100:>8.2f}%    {improved_summary['mean_ar_news']*100:>8.2f}%    {(improved_summary['mean_ar_news'] - previous_summary['mean_ar_news'])*100:>+8.2f}%")
                print(f"  Mean AR (Non-Events)     {previous_summary['mean_ar_non_news']*100:>8.2f}%    {improved_summary['mean_ar_non_news']*100:>8.2f}%    {(improved_summary['mean_ar_non_news'] - previous_summary['mean_ar_non_news'])*100:>+8.2f}%")
                print(f"  Significant Tests        {previous_summary['significant_tests']:>8}    {improved_summary['significant_tests']:>8}    {improved_summary['significant_tests'] - previous_summary['significant_tests']:>+8}")
                print(f"  Model R²                 {previous_summary['avg_r_squared']:>8.3f}    {improved_summary['avg_r_squared']:>8.3f}    {improved_summary['avg_r_squared'] - previous_summary['avg_r_squared']:>+8.3f}")
                
                # Calculate impact magnitude
                prev_impact = abs(previous_summary['mean_ar_news'] - previous_summary['mean_ar_non_news']) * 100
                impr_impact = abs(improved_summary['mean_ar_news'] - improved_summary['mean_ar_non_news']) * 100
                
                print(f"\n🎯 IMPACT COMPARISON:")
                print(f"  Previous Impact: {prev_impact:.3f}%")
                print(f"  Improved Impact: {impr_impact:.3f}%")
                print(f"  Change: {impr_impact - prev_impact:+.3f}%")
                
                if impr_impact > prev_impact:
                    print(f"  ✅ Improvement: Better impact magnitude!")
                else:
                    print(f"  ⚠️  No improvement in impact magnitude")
    
    def run_complete_improved_study(self):
        """Run complete improved event study"""
        print(f"\n{'='*80}")
        print(f"IMPROVED EVENT STUDY ANALYSIS")
        print(f"Using Strategy A: Rare High-Impact Events")
        print(f"{'='*80}")
        
        results = {}
        
        # Process AAPL
        print(f"\n{'🍎'*40}")
        results['AAPL'] = self.run_improved_event_study('AAPL')
        
        # Process TSLA
        print(f"\n{'⚡'*40}")
        results['TSLA'] = self.run_improved_event_study('TSLA')
        
        # Compare with previous results
        for ticker in ['AAPL', 'TSLA']:
            if results[ticker]:
                self.compare_with_previous(ticker)
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"IMPROVED EVENT STUDY SUMMARY")
        print(f"{'='*80}")
        
        for ticker, summary in results.items():
            if summary:
                print(f"\n{ticker} IMPROVED EVENT STUDY:")
                print(f"  Event Days: {summary['news_days']}")
                print(f"  Mean AR (Events): {summary['mean_ar_news']*100:.2f}%")
                print(f"  Mean AR (Non-Events): {summary['mean_ar_non_news']*100:.2f}%")
                print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
                print(f"  Model R²: {summary['avg_r_squared']:.3f}")
                
                # Calculate impact magnitude
                impact = abs(summary['mean_ar_news'] - summary['mean_ar_non_news']) * 100
                
                if summary['significant_tests'] >= 2 and impact > 1.0:
                    print(f"  🎉 SUCCESS: High-impact events show significant impact!")
                elif summary['significant_tests'] >= 1 and impact > 0.5:
                    print(f"  ✅ PARTIAL: Some evidence of event impact")
                elif impact > 1.0:
                    print(f"  ⚠️  ECONOMIC: Large impact but not significant")
                else:
                    print(f"  ❌ NO IMPACT: Events don't show meaningful impact")
            else:
                print(f"\n{ticker}: ❌ Improved event study failed")
        
        print(f"\n{'='*80}")
        print(f"🎉 IMPROVED EVENT STUDY COMPLETE!")
        print(f"{'='*80}")


def main():
    """Run the improved event study"""
    event_study = ImprovedEventStudy()
    event_study.run_complete_improved_study()


if __name__ == "__main__":
    main()
