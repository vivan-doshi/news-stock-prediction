"""
Run Analysis with Filtered News Data
Applies the recommended filtering pipeline and runs the complete analysis
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


class FilteredPhase1Analysis(Phase1Analysis):
    """Enhanced Phase1Analysis with filtered news data"""
    
    def __init__(self, stock_file, news_file, ff_file, sector_file=None, 
                 data_dir="../01-data", output_dir="../03-output"):
        super().__init__(stock_file, news_file, ff_file, sector_file, data_dir, output_dir)
        
    def apply_news_filtering(self, raw_news_file, filtered_news_file):
        """
        Apply the recommended filtering pipeline to news data
        
        Parameters:
        -----------
        raw_news_file : str
            Path to raw EODHD news file
        filtered_news_file : str
            Path to save filtered news file
        """
        print(f"\n[Filtering] Applying news filtering pipeline...")
        
        # Load raw news data
        raw_news_path = Path(self.data_dir) / raw_news_file
        df = pd.read_csv(raw_news_path)
        print(f"  ✓ Loaded {len(df):,} raw articles")
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Step 1: Remove articles with empty tags
        def parse_tags(tag_str):
            try:
                if pd.isna(tag_str) or tag_str == '[]' or tag_str == '':
                    return []
                import ast
                return ast.literal_eval(tag_str)
            except:
                return []
        
        df['tags_list'] = df['tags'].apply(parse_tags)
        has_tags = df['tags_list'].apply(len) > 0
        df_filtered = df[has_tags].copy()
        print(f"  ✓ Step 1: Removed empty tags - {len(df_filtered):,} remaining ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Step 2: Keep only company-specific articles
        company_tags = ['APPLE', 'TESLA', 'ELON MUSK', 'IPHONE', 'APPLE WATCH', 'APPLE PAY', 
                       'MODEL S', 'MODEL 3', 'MODEL X', 'MODEL Y', 'CYBERTRUCK', 'TESLA STOCK',
                       'APPLE STOCK', 'APPLE SHARES', 'TESLA SHARES']
        
        company_specific = df_filtered['tags_list'].apply(
            lambda tags: any(any(company_tag.upper() in tag.upper() for company_tag in company_tags) 
                            for tag in tags) if isinstance(tags, list) else False
        )
        df_filtered = df_filtered[company_specific].copy()
        print(f"  ✓ Step 2: Company-specific only - {len(df_filtered):,} remaining ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Step 3: Remove very short content (<200 chars)
        df_filtered['content_length'] = df_filtered['content'].str.len()
        substantial_content = df_filtered['content_length'] >= 200
        df_filtered = df_filtered[substantial_content].copy()
        print(f"  ✓ Step 3: Substantial content only - {len(df_filtered):,} remaining ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Step 4: Remove duplicate titles
        no_duplicates = ~df_filtered.duplicated(subset=['title'], keep='first')
        df_filtered = df_filtered[no_duplicates].copy()
        print(f"  ✓ Step 4: No duplicate titles - {len(df_filtered):,} remaining ({len(df_filtered)/len(df)*100:.1f}%)")
        
        # Create news dates file from filtered data
        news_dates = pd.DataFrame({'Date': df_filtered['date'].dt.date.unique()})
        news_dates = news_dates.sort_values('Date')
        
        # Save filtered news dates
        output_path = Path(self.data_dir) / filtered_news_file
        news_dates.to_csv(output_path, index=False)
        
        print(f"  ✓ Created {filtered_news_file} with {len(news_dates)} unique news dates")
        print(f"  📊 Filtering Summary: {len(df):,} → {len(df_filtered):,} articles ({len(df_filtered)/len(df)*100:.1f}% retained)")
        
        return len(df_filtered), len(news_dates)
    
    def run_filtered_analysis(self, raw_news_file):
        """
        Run complete analysis with filtered news data
        
        Parameters:
        -----------
        raw_news_file : str
            Raw EODHD news file name
        """
        print("=" * 70)
        print("FILTERED NEWS-STOCK ANALYSIS")
        print("=" * 70)
        
        # Create filtered news file name
        filtered_news_file = raw_news_file.replace('_eodhd_news.csv', '_news_dates_filtered.csv')
        
        # Apply filtering
        filtered_articles, filtered_dates = self.apply_news_filtering(raw_news_file, filtered_news_file)
        
        if filtered_articles == 0:
            print("❌ No articles remaining after filtering!")
            return None
        
        # Update news file to use filtered data
        original_news_file = self.news_file
        self.news_file = filtered_news_file
        
        # Run the standard analysis
        print(f"\n[Analysis] Running analysis with {filtered_dates:,} filtered news dates...")
        summary = self.run_complete_analysis()
        
        # Restore original news file name
        self.news_file = original_news_file
        
        # Add filtering info to summary
        if summary:
            summary['filtered_articles'] = filtered_articles
            summary['filtered_dates'] = filtered_dates
        
        return summary


def run_filtered_analysis_for_ticker(ticker, start_date, end_date):
    """
    Run filtered analysis for a single ticker
    """
    print("\n" + "=" * 80)
    print(f"RUNNING FILTERED ANALYSIS FOR {ticker}")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    
    # File names
    stock_file = f"{ticker}_stock_data.csv"
    raw_news_file = f"{ticker}_eodhd_news.csv"
    ff_file = "fama_french_factors.csv"
    output_dir = f"../03-output/{ticker}_filtered"
    
    try:
        analysis = FilteredPhase1Analysis(
            stock_file=stock_file,
            news_file="",  # Will be set by filtering
            ff_file=ff_file,
            sector_file=None,
            data_dir="../01-data",
            output_dir=output_dir
        )
        
        summary = analysis.run_filtered_analysis(raw_news_file)
        
        if summary:
            print(f"\n✅ {ticker} Filtered Analysis Complete!")
            print(f"   Output saved to: {Path(output_dir).absolute()}")
            return summary
        else:
            print(f"\n❌ {ticker} Filtered Analysis Failed!")
            return None
            
    except Exception as e:
        print(f"\n❌ {ticker} Filtered Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run filtered analysis for both AAPL and TSLA"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "FILTERED NEWS-STOCK ANALYSIS")
    print("=" * 80)
    print("\nApplying recommended filtering pipeline:")
    print("  Step 1: Remove articles with empty tags")
    print("  Step 2: Keep only company-specific articles")
    print("  Step 3: Remove very short content (<200 chars)")
    print("  Step 4: Remove duplicate titles")
    print("\nThen running Phase 1 event study analysis...")
    print("=" * 80)
    
    results = {}
    
    # Analyze AAPL
    print("\n\n" + "🍎" * 40)
    results['AAPL'] = run_filtered_analysis_for_ticker(
        ticker='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-31'
    )
    
    # Analyze TSLA
    print("\n\n" + "⚡" * 40)
    results['TSLA'] = run_filtered_analysis_for_ticker(
        ticker='TSLA',
        start_date='2020-01-01',
        end_date='2025-10-08'
    )
    
    # Final Summary
    print("\n" + "=" * 80)
    print(" " * 30 + "FILTERED ANALYSIS SUMMARY")
    print("=" * 80)
    
    for ticker, summary in results.items():
        if summary:
            print(f"\n{ticker} (FILTERED):")
            print(f"  Filtered Articles: {summary.get('filtered_articles', 'N/A'):,}")
            print(f"  Filtered News Dates: {summary.get('filtered_dates', 'N/A'):,}")
            print(f"  News Days: {summary['news_days']}")
            print(f"  Mean AR (News): {summary['mean_ar_news']*100:.2f}%")
            print(f"  Mean AR (Non-News): {summary['mean_ar_non_news']*100:.2f}%")
            print(f"  Significant Tests: {summary['significant_tests']}/{summary['total_tests']}")
            print(f"  Model R²: {summary['avg_r_squared']:.3f}")
            
            # Success evaluation
            if summary['significant_tests'] >= 3:
                print(f"  ✅ Strong evidence of news impact!")
            elif summary['significant_tests'] >= 2:
                print(f"  ⚠️  Some evidence of news impact")
            else:
                print(f"  ❌ Weak evidence of news impact")
        else:
            print(f"\n{ticker}: ❌ Analysis failed")
    
    print("\n" + "=" * 80)
    print("🎉 FILTERED ANALYSES COMPLETE!")
    print("=" * 80)
    print("\n📁 Results saved to:")
    print("   - 03-output/AAPL_filtered/")
    print("   - 03-output/TSLA_filtered/")
    print("\nKey output files:")
    print("   - abnormal_returns.csv")
    print("   - beta_estimates.csv")
    print("   - statistical_tests.csv")
    print("   - analysis_summary.png")
    print("   - analysis_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
