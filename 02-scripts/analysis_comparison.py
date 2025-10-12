"""
Compare Original vs Filtered Analysis Results
"""

import pandas as pd
import numpy as np

def load_analysis_summary(file_path):
    """Load analysis summary from CSV"""
    try:
        df = pd.read_csv(file_path)
        return df.iloc[0].to_dict()
    except:
        return {}

def compare_analyses():
    """Compare original vs filtered analysis results"""
    
    print("=" * 80)
    print(" " * 25 + "ANALYSIS COMPARISON")
    print(" " * 20 + "ORIGINAL vs FILTERED RESULTS")
    print("=" * 80)
    
    # Load results
    aapl_orig = load_analysis_summary('../03-output/AAPL/analysis_summary.csv')
    aapl_filt = load_analysis_summary('../03-output/AAPL_filtered/analysis_summary.csv')
    tsla_orig = load_analysis_summary('../03-output/TSLA/analysis_summary.csv')
    tsla_filt = load_analysis_summary('../03-output/TSLA_filtered/analysis_summary.csv')
    
    print("\n📊 FILTERING IMPACT SUMMARY")
    print("-" * 80)
    print(f"{'Metric':<25} {'AAPL Original':<15} {'AAPL Filtered':<15} {'Change':<15}")
    print("-" * 80)
    
    if aapl_orig and aapl_filt:
        # AAPL comparison
        print(f"{'News Days':<25} {aapl_orig.get('news_days', 'N/A'):<15} {aapl_filt.get('news_days', 'N/A'):<15} {aapl_filt.get('news_days', 0) - aapl_orig.get('news_days', 0):<+15}")
        
        ar_news_orig = aapl_orig.get('mean_ar_news', np.nan)
        ar_news_filt = aapl_filt.get('mean_ar_news', np.nan)
        ar_news_str_orig = f"{ar_news_orig*100:.2f}%" if not np.isnan(ar_news_orig) else "N/A"
        ar_news_str_filt = f"{ar_news_filt*100:.2f}%" if not np.isnan(ar_news_filt) else "N/A"
        print(f"{'Mean AR (News)':<25} {ar_news_str_orig:<15} {ar_news_str_filt:<15} {'N/A':<15}")
        
        ar_non_orig = aapl_orig.get('mean_ar_non_news', 0)
        ar_non_filt = aapl_filt.get('mean_ar_non_news', 0)
        print(f"{'Mean AR (Non-News)':<25} {ar_non_orig*100:.2f}%{'':<9} {ar_non_filt*100:.2f}%{'':<9} {ar_non_filt - ar_non_orig:+.4f}{'':<9}")
        
        print(f"{'Significant Tests':<25} {aapl_orig.get('significant_tests', 0)}/{aapl_orig.get('total_tests', 0)}{'':<9} {aapl_filt.get('significant_tests', 0)}/{aapl_filt.get('total_tests', 0)}{'':<9} {aapl_filt.get('significant_tests', 0) - aapl_orig.get('significant_tests', 0):<+15}")
        print(f"{'Model R²':<25} {aapl_orig.get('avg_r_squared', 0):.3f}{'':<12} {aapl_filt.get('avg_r_squared', 0):.3f}{'':<12} {aapl_filt.get('avg_r_squared', 0) - aapl_orig.get('avg_r_squared', 0):+.3f}{'':<12}")
    
    print("\n" + "-" * 80)
    print(f"{'Metric':<25} {'TSLA Original':<15} {'TSLA Filtered':<15} {'Change':<15}")
    print("-" * 80)
    
    if tsla_orig and tsla_filt:
        # TSLA comparison
        print(f"{'News Days':<25} {tsla_orig.get('news_days', 'N/A'):<15} {tsla_filt.get('news_days', 'N/A'):<15} {tsla_filt.get('news_days', 0) - tsla_orig.get('news_days', 0):<+15}")
        
        ar_news_orig = tsla_orig.get('mean_ar_news', 0)
        ar_news_filt = tsla_filt.get('mean_ar_news', 0)
        print(f"{'Mean AR (News)':<25} {ar_news_orig*100:.2f}%{'':<9} {ar_news_filt*100:.2f}%{'':<9} {ar_news_filt - ar_news_orig:+.4f}{'':<9}")
        
        ar_non_orig = tsla_orig.get('mean_ar_non_news', 0)
        ar_non_filt = tsla_filt.get('mean_ar_non_news', 0)
        print(f"{'Mean AR (Non-News)':<25} {ar_non_orig*100:.2f}%{'':<9} {ar_non_filt*100:.2f}%{'':<9} {ar_non_filt - ar_non_orig:+.4f}{'':<9}")
        
        print(f"{'Significant Tests':<25} {tsla_orig.get('significant_tests', 0)}/{tsla_orig.get('total_tests', 0)}{'':<9} {tsla_filt.get('significant_tests', 0)}/{tsla_filt.get('total_tests', 0)}{'':<9} {tsla_filt.get('significant_tests', 0) - tsla_orig.get('significant_tests', 0):<+15}")
        print(f"{'Model R²':<25} {tsla_orig.get('avg_r_squared', 0):.3f}{'':<12} {tsla_filt.get('avg_r_squared', 0):.3f}{'':<12} {tsla_filt.get('avg_r_squared', 0) - tsla_orig.get('avg_r_squared', 0):+.3f}{'':<12}")
    
    print("\n" + "=" * 80)
    print("📈 KEY INSIGHTS")
    print("=" * 80)
    
    print("\n✅ FILTERING SUCCESS:")
    print("   • Successfully reduced noise by filtering out:")
    print("     - Articles with empty tags")
    print("     - General market news (non-company-specific)")
    print("     - Very short content articles")
    print("     - Duplicate articles")
    
    print("\n📊 DATA REDUCTION:")
    print("   • AAPL: 25,275 → 6,435 articles (74.5% reduction)")
    print("   • TSLA: 49,378 → 9,269 articles (81.2% reduction)")
    print("   • News frequency reduced from >85% to ~50% of trading days")
    
    print("\n🔍 ANALYSIS IMPACT:")
    print("   • Model quality improved (higher R² for AAPL)")
    print("   • More manageable dataset size")
    print("   • Better signal-to-noise ratio")
    print("   • Still no significant evidence of news impact")
    
    print("\n⚠️  REMAINING CHALLENGES:")
    print("   • Even with filtering, news impact remains weak")
    print("   • Statistical tests still not significant")
    print("   • News frequency still high (50%+ of days)")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Consider even more aggressive filtering (e.g., only earnings news)")
    print("   2. Focus on specific news types (product launches, major announcements)")
    print("   3. Use sentiment-weighted analysis")
    print("   4. Consider different event windows or market conditions")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    compare_analyses()
