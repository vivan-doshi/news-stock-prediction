"""
Run Analysis Automatically (No Input Required)
"""

from run_analysis import analyze_multiple_stocks

if __name__ == "__main__":
    print("\n🚀 Starting Multi-Stock Analysis Pipeline (Auto Mode)...\n")
    print("Analyzing AAPL and TSLA with Marketaux news data...")

    results = analyze_multiple_stocks()
    print("\n✅ Analysis complete!")
