"""
Debug script to investigate the NaN issue in event study
"""

import pandas as pd
import numpy as np
from pathlib import Path
import importlib

# Import data loader
data_loader_module = importlib.import_module('01_data_loader')
load_all_data = data_loader_module.load_all_data

def debug_data_loading():
    """Debug the data loading process"""
    print("="*80)
    print("DEBUGGING DATA LOADING")
    print("="*80)

    data_dir = "../01-data"
    filtered_dir = Path("../03-output/filtered_analysis")

    # Test with TSLA
    ticker = "TSLA"
    stock_file = f"{ticker}_stock_data.csv"
    event_dates_file = str(filtered_dir / f"{ticker}_event_dates.csv")
    ff_file = "fama_french_factors.csv"

    print(f"\n1. Loading stock data: {stock_file}")
    stock_df = pd.read_csv(Path(data_dir) / stock_file)
    print(f"   Shape: {stock_df.shape}")
    print(f"   Columns: {stock_df.columns.tolist()}")
    print(f"   Date range: {stock_df['Date'].min()} to {stock_df['Date'].max()}")
    print(f"   Sample:\n{stock_df.head()}")

    print(f"\n2. Loading Fama-French factors: {ff_file}")
    ff_df = pd.read_csv(Path(data_dir) / ff_file)
    print(f"   Shape: {ff_df.shape}")
    print(f"   Columns: {ff_df.columns.tolist()}")
    print(f"   Date range: {ff_df['Date'].min()} to {ff_df['Date'].max()}")
    print(f"   Sample:\n{ff_df.head()}")

    print(f"\n3. Loading event dates: {event_dates_file}")
    event_df = pd.read_csv(event_dates_file)
    print(f"   Shape: {event_df.shape}")
    print(f"   Columns: {event_df.columns.tolist()}")
    print(f"   Date range: {event_df['Date'].min()} to {event_df['Date'].max()}")
    print(f"   Sample:\n{event_df.head()}")

    print(f"\n4. Testing load_all_data function...")
    try:
        merged_data, news_dates = load_all_data(
            stock_file=stock_file,
            news_file=event_dates_file,
            ff_file=ff_file,
            sector_file=None,
            data_dir=data_dir
        )

        print(f"   ✓ Merged data shape: {merged_data.shape}")
        print(f"   ✓ Merged data columns: {merged_data.columns.tolist()}")
        print(f"   ✓ Date range: {merged_data.index.min()} to {merged_data.index.max()}")
        print(f"   ✓ News dates count: {len(news_dates)}")
        print(f"\n   Merged data sample:")
        print(merged_data.head())

        print(f"\n5. Checking for NaN values in merged data:")
        nan_counts = merged_data.isnull().sum()
        print(nan_counts)

        print(f"\n6. Checking data statistics:")
        print(merged_data.describe())

        # Check if there's actually usable data
        print(f"\n7. Checking Excess_Return:")
        print(f"   Mean: {merged_data['Excess_Return'].mean():.6f}")
        print(f"   Std: {merged_data['Excess_Return'].std():.6f}")
        print(f"   Non-null count: {merged_data['Excess_Return'].notna().sum()}")

        print(f"\n8. Checking factor columns:")
        factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        for col in factor_cols:
            if col in merged_data.columns:
                print(f"   {col}: mean={merged_data[col].mean():.6f}, non-null={merged_data[col].notna().sum()}")

        return merged_data, news_dates

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def debug_beta_estimation(merged_data, news_dates):
    """Debug beta estimation"""
    print("\n" + "="*80)
    print("DEBUGGING BETA ESTIMATION")
    print("="*80)

    if merged_data is None:
        print("Skipping - no merged data available")
        return

    from importlib import import_module
    beta_module = import_module('02_beta_estimation')
    BetaEstimator = beta_module.BetaEstimator

    print(f"\n1. Creating BetaEstimator...")
    estimator = BetaEstimator(window_size=126, min_periods=100)

    print(f"\n2. Running rolling beta estimation...")
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    try:
        beta_df = estimator.rolling_beta_estimation(
            data=merged_data,
            factor_cols=factor_cols,
            exclude_dates=news_dates,
            event_window=(-1, 2)
        )

        print(f"   ✓ Beta estimation complete")
        print(f"   Shape: {beta_df.shape}")
        print(f"   Columns: {beta_df.columns.tolist()}")

        print(f"\n3. Checking beta results:")
        print(beta_df.head(10))

        print(f"\n4. Checking for NaN values in betas:")
        nan_counts = beta_df.isnull().sum()
        print(nan_counts)

        print(f"\n5. R-squared statistics:")
        print(f"   Mean: {beta_df['R_squared'].mean()}")
        print(f"   Non-null count: {beta_df['R_squared'].notna().sum()}")

        return beta_df

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def debug_abnormal_returns(merged_data, beta_df, news_dates):
    """Debug abnormal returns calculation"""
    print("\n" + "="*80)
    print("DEBUGGING ABNORMAL RETURNS CALCULATION")
    print("="*80)

    if merged_data is None or beta_df is None:
        print("Skipping - no data available")
        return

    from importlib import import_module
    ar_module = import_module('03_abnormal_returns')
    AbnormalReturnsCalculator = ar_module.AbnormalReturnsCalculator

    print(f"\n1. Creating AbnormalReturnsCalculator...")
    calculator = AbnormalReturnsCalculator()

    print(f"\n2. Calculating abnormal returns...")
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

    try:
        ar_df = calculator.calculate_abnormal_returns(
            data=merged_data,
            beta_df=beta_df,
            factor_cols=factor_cols,
            news_dates=news_dates
        )

        print(f"   ✓ Abnormal returns calculation complete")
        print(f"   Shape: {ar_df.shape}")
        print(f"   Columns: {ar_df.columns.tolist()}")

        print(f"\n3. Checking AR results:")
        print(ar_df.head(10))

        print(f"\n4. Checking for NaN values in AR:")
        nan_counts = ar_df.isnull().sum()
        print(nan_counts)

        print(f"\n5. AR statistics (excluding NaN):")
        print(f"   Mean: {ar_df['Abnormal_Return'].mean()}")
        print(f"   Std: {ar_df['Abnormal_Return'].std()}")
        print(f"   Non-null count: {ar_df['Abnormal_Return'].notna().sum()}")

        return ar_df

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all debug checks"""
    print("\n" + "#"*80)
    print("EVENT STUDY DEBUG SCRIPT")
    print("#"*80)

    # Debug data loading
    merged_data, news_dates = debug_data_loading()

    # Debug beta estimation
    beta_df = debug_beta_estimation(merged_data, news_dates)

    # Debug abnormal returns
    ar_df = debug_abnormal_returns(merged_data, beta_df, news_dates)

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

    if ar_df is not None and ar_df['Abnormal_Return'].notna().sum() > 0:
        print("✅ Analysis pipeline is working correctly")
    else:
        print("❌ Issue detected - AR calculation producing NaN values")


if __name__ == "__main__":
    main()
