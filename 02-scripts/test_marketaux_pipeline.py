"""
Test Marketaux Pipeline with Limited Data
"""

from run_complete_pipeline import CompletePipeline
from pathlib import Path

# Test with AAPL and short date range (Jan 2024 only)
ticker = 'AAPL'
start_date = '2024-01-01'
end_date = '2024-01-31'
sector_etf = 'XLK'

print(f"\n{'='*70}")
print("TESTING MARKETAUX PIPELINE")
print(f"{'='*70}")
print(f"\nTicker: {ticker}")
print(f"Period: {start_date} to {end_date}")
print(f"Sector: {sector_etf}")
print(f"{'='*70}\n")

# Create output directory
output_dir = f"../03-output/{ticker}"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Run pipeline
pipeline = CompletePipeline(
    ticker=ticker,
    start_date=start_date,
    end_date=end_date,
    sector_ticker=sector_etf,
    marketaux_api_key=None,  # Loads from .env
    data_dir="../01-data",
    output_dir=output_dir
)

pipeline.run_full_pipeline(skip_data_download=False)

print(f"\n{'='*70}")
print("TEST COMPLETE!")
print(f"Results saved to: {output_dir}")
print(f"{'='*70}")
