"""
COMPREHENSIVE 17-STOCK EXPERIMENT
==================================

Complete analysis with 17 stocks across 10 sectors:
- Data validation
- Feature engineering
- Baseline models
- News prediction models
- Sector analysis
- Results report
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_DIR = '01-data'
OUTPUT_DIR = '03-output/comprehensive_experiment'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 17 stocks with complete data
STOCKS_WITH_DATA = {
    'AAPL': {'sector': 'Technology', 'etf': 'XLK'},
    'MSFT': {'sector': 'Technology', 'etf': 'XLK'},
    'NVDA': {'sector': 'Technology', 'etf': 'XLK'},
    'JPM': {'sector': 'Finance', 'etf': 'XLF'},
    'GS': {'sector': 'Finance', 'etf': 'XLF'},
    'JNJ': {'sector': 'Healthcare', 'etf': 'XLV'},
    'PFE': {'sector': 'Healthcare', 'etf': 'XLV'},
    'AMZN': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'TSLA': {'sector': 'Consumer Discretionary', 'etf': 'XLY'},
    'PG': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'WMT': {'sector': 'Consumer Staples', 'etf': 'XLP'},
    'GOOGL': {'sector': 'Communication Services', 'etf': 'XLC'},
    'META': {'sector': 'Communication Services', 'etf': 'XLC'},
    'XOM': {'sector': 'Energy', 'etf': 'XLE'},
    'BA': {'sector': 'Industrials', 'etf': 'XLI'},
    'NEE': {'sector': 'Utilities', 'etf': 'XLU'},
    'AMT': {'sector': 'Real Estate', 'etf': 'XLRE'}
}

# Train/test split dates (adjusted for data availability)
TRAIN_START = '2020-01-01'
TRAIN_END = '2023-06-30'
TEST_START = '2023-07-01'
TEST_END = '2024-01-31'

print("=" * 100)
print("COMPREHENSIVE 17-STOCK EXPERIMENT")
print("=" * 100)
print(f"\nStocks: {len(STOCKS_WITH_DATA)}")
print(f"Sectors: {len(set([s['sector'] for s in STOCKS_WITH_DATA.values()]))}")
print(f"\nTrain period: {TRAIN_START} to {TRAIN_END}")
print(f"Test period: {TEST_START} to {TEST_END}")

# ============================================================================
# STEP 1: DATA VALIDATION
# ============================================================================

print("\n" + "=" * 100)
print("STEP 1: DATA VALIDATION")
print("=" * 100)

validation_results = []

for ticker, info in STOCKS_WITH_DATA.items():
    stock_file = os.path.join(DATA_DIR, f'{ticker}_stock_data.csv')
    news_file = os.path.join(DATA_DIR, f'{ticker}_eodhd_news.csv')

    result = {'ticker': ticker, 'sector': info['sector']}

    # Check stock data
    if os.path.exists(stock_file):
        stock_df = pd.read_csv(stock_file)
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        result['stock_records'] = len(stock_df)
        result['stock_date_min'] = stock_df['Date'].min()
        result['stock_date_max'] = stock_df['Date'].max()
        result['stock_ok'] = True
    else:
        result['stock_ok'] = False

    # Check news data
    if os.path.exists(news_file):
        news_df = pd.read_csv(news_file)
        result['news_records'] = len(news_df)
        result['news_ok'] = True
    else:
        result['news_ok'] = False

    validation_results.append(result)

validation_df = pd.DataFrame(validation_results)
print(f"\n✓ All stocks validated:")
print(validation_df[['ticker', 'sector', 'stock_records', 'news_records']])

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 100)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 100)

all_features = []

for ticker, info in STOCKS_WITH_DATA.items():
    print(f"\nProcessing {ticker} ({info['sector']})...")

    # Load stock data
    stock_df = pd.read_csv(os.path.join(DATA_DIR, f'{ticker}_stock_data.csv'))
    stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True).dt.tz_localize(None)
    stock_df = stock_df.sort_values('Date').reset_index(drop=True)

    # Handle different column names (old vs new data)
    price_col = 'Adjusted_close' if 'Adjusted_close' in stock_df.columns else 'Close'

    # Calculate returns
    stock_df['returns'] = stock_df[price_col].pct_change()
    stock_df['returns_next'] = stock_df['returns'].shift(-1)  # Target variable

    # Price features
    stock_df['volatility_5d'] = stock_df['returns'].rolling(5).std()
    stock_df['volatility_20d'] = stock_df['returns'].rolling(20).std()
    stock_df['momentum_5d'] = stock_df[price_col].pct_change(5)
    stock_df['momentum_20d'] = stock_df[price_col].pct_change(20)

    # Volume features
    stock_df['volume_ma_20'] = stock_df['Volume'].rolling(20).mean()
    stock_df['volume_ratio'] = stock_df['Volume'] / stock_df['volume_ma_20']

    # Load and process news data
    news_df = pd.read_csv(os.path.join(DATA_DIR, f'{ticker}_eodhd_news.csv'))
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.sort_values('date')

    # Aggregate news by date
    news_df['date_only'] = pd.to_datetime(news_df['date']).dt.date
    news_agg = news_df.groupby('date_only').agg({
        'sentiment_polarity': ['mean', 'std', 'count'],
        'sentiment_pos': 'mean',
        'sentiment_neg': 'mean'
    }).reset_index()

    news_agg.columns = ['date', 'sent_mean', 'sent_std', 'news_count', 'sent_pos', 'sent_neg']
    news_agg['date'] = pd.to_datetime(news_agg['date'])

    # Merge with stock data (ensure both are datetime)
    stock_df['Date_normalized'] = pd.to_datetime(stock_df['Date']).dt.normalize()
    news_agg['date_normalized'] = pd.to_datetime(news_agg['date']).dt.normalize()
    stock_df = stock_df.merge(news_agg, left_on='Date_normalized', right_on='date_normalized', how='left')

    # Drop extra columns from merge
    stock_df = stock_df.drop(['date', 'date_normalized', 'Date_normalized'], axis=1, errors='ignore')

    # Fill missing news values with 0
    news_cols = ['sent_mean', 'sent_std', 'news_count', 'sent_pos', 'sent_neg']
    stock_df[news_cols] = stock_df[news_cols].fillna(0)

    # Create rolling news features
    for window in [1, 3, 7]:
        stock_df[f'sent_mean_{window}d'] = stock_df['sent_mean'].rolling(window, min_periods=1).mean()
        stock_df[f'news_count_{window}d'] = stock_df['news_count'].rolling(window, min_periods=1).sum()

    # Add metadata
    stock_df['ticker'] = ticker
    stock_df['sector'] = info['sector']

    # Keep only the columns we need (drop extra columns like Dividends, Stock Splits)
    keep_cols = ['Date', 'returns', 'returns_next', 'volatility_5d', 'volatility_20d',
                 'momentum_5d', 'momentum_20d', 'volume_ratio',
                 'sent_mean', 'sent_std', 'news_count', 'sent_pos', 'sent_neg',
                 'sent_mean_1d', 'sent_mean_3d', 'sent_mean_7d',
                 'news_count_1d', 'news_count_3d', 'news_count_7d',
                 'ticker', 'sector']
    stock_df = stock_df[keep_cols]

    all_features.append(stock_df)
    print(f"  ✓ {len(stock_df)} records with {stock_df[news_cols].sum().sum():.0f} total news items")

# Combine all data
combined_df = pd.concat(all_features, ignore_index=True)

print(f"\n✓ Combined dataset before filtering: {len(combined_df)} records")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "=" * 100)
print("STEP 3: TRAIN/TEST SPLIT")
print("=" * 100)

combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Filter by date first
train_df = combined_df[(combined_df['Date'] >= TRAIN_START) & (combined_df['Date'] <= TRAIN_END)].copy()
test_df = combined_df[(combined_df['Date'] >= TEST_START) & (combined_df['Date'] <= TEST_END)].copy()

print(f"\nBefore dropping NaN:")
print(f"  Train set: {len(train_df)} records")
print(f"  Test set: {len(test_df)} records")

# Drop rows with missing values AFTER date split
train_df = train_df.dropna()
test_df = test_df.dropna()

print(f"\nTrain set: {len(train_df)} records ({TRAIN_START} to {TRAIN_END})")
print(f"Test set: {len(test_df)} records ({TEST_START} to {TEST_END})")

# Define feature sets
price_features = ['returns', 'volatility_5d', 'volatility_20d', 'momentum_5d', 'momentum_20d', 'volume_ratio']
news_features = ['sent_mean', 'sent_std', 'news_count', 'sent_pos', 'sent_neg',
                 'sent_mean_1d', 'sent_mean_3d', 'sent_mean_7d',
                 'news_count_1d', 'news_count_3d', 'news_count_7d']
all_predictors = price_features + news_features

target = 'returns_next'

# ============================================================================
# STEP 4: BASELINE MODELS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 4: BASELINE MODELS")
print("=" * 100)

results = []

# Baseline 1: Random (mean return)
train_mean = train_df[target].mean()
test_pred_random = np.full(len(test_df), train_mean)

mse_random = mean_squared_error(test_df[target], test_pred_random)
mae_random = mean_absolute_error(test_df[target], test_pred_random)

print(f"\n1. Random Baseline (mean return)")
print(f"   MSE: {mse_random:.6f}")
print(f"   MAE: {mae_random:.6f}")

results.append({
    'model': 'Random',
    'features': 'none',
    'mse': mse_random,
    'mae': mae_random,
    'r2': 0.0
})

# Baseline 2: Momentum (previous return predicts next)
momentum_pred = test_df['returns'].values
mse_momentum = mean_squared_error(test_df[target], momentum_pred)
mae_momentum = mean_absolute_error(test_df[target], momentum_pred)
r2_momentum = r2_score(test_df[target], momentum_pred)

print(f"\n2. Momentum Baseline (previous return)")
print(f"   MSE: {mse_momentum:.6f}")
print(f"   MAE: {mae_momentum:.6f}")
print(f"   R²: {r2_momentum:.6f}")

results.append({
    'model': 'Momentum',
    'features': 'price',
    'mse': mse_momentum,
    'mae': mae_momentum,
    'r2': r2_momentum
})

# ============================================================================
# STEP 5: PREDICTION MODELS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 5: PREDICTION MODELS")
print("=" * 100)

models_to_test = [
    ('Linear (Price)', LinearRegression(), price_features),
    ('Linear (News)', LinearRegression(), news_features),
    ('Linear (All)', LinearRegression(), all_predictors),
    ('Ridge (All)', Ridge(alpha=1.0), all_predictors),
    ('RandomForest (All)', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42), all_predictors),
]

for model_name, model, features in models_to_test:
    print(f"\n{model_name}:")

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   R²: {r2:.6f}")

    # Direction accuracy
    direction_correct = ((y_pred > 0) == (y_test > 0)).mean()
    print(f"   Direction Accuracy: {direction_correct:.1%}")

    results.append({
        'model': model_name,
        'features': '/'.join(features[:2]) if len(features) > 2 else features[0] if len(features) == 1 else 'multiple',
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'direction_acc': direction_correct
    })

# ============================================================================
# STEP 6: RESULTS SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("STEP 6: RESULTS SUMMARY")
print("=" * 100)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mae')

print("\nModel Performance Comparison (sorted by MAE):")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results.csv'), index=False)
print(f"\n✓ Results saved to: {OUTPUT_DIR}/model_results.csv")

# ============================================================================
# STEP 7: SECTOR ANALYSIS
# ============================================================================

print("\n" + "=" * 100)
print("STEP 7: SECTOR ANALYSIS")
print("=" * 100)

sector_results = []

for sector in test_df['sector'].unique():
    sector_data = test_df[test_df['sector'] == sector]

    if len(sector_data) < 10:
        continue

    # Calculate sector statistics
    avg_return = sector_data[target].mean()
    volatility = sector_data[target].std()
    news_volume = sector_data['news_count'].mean()

    sector_results.append({
        'sector': sector,
        'avg_return': avg_return,
        'volatility': volatility,
        'news_volume': news_volume,
        'n_observations': len(sector_data)
    })

sector_df = pd.DataFrame(sector_results)
print("\nSector Statistics:")
print(sector_df.to_string(index=False))

sector_df.to_csv(os.path.join(OUTPUT_DIR, 'sector_analysis.csv'), index=False)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 100)
print("EXPERIMENT COMPLETE!")
print("=" * 100)

print(f"\n✓ Analyzed {len(STOCKS_WITH_DATA)} stocks across {len(sector_df)} sectors")
print(f"✓ Tested {len(results_df)} models")
print(f"✓ Best model: {results_df.iloc[0]['model']} (MAE: {results_df.iloc[0]['mae']:.6f})")

best_vs_random = (1 - results_df.iloc[0]['mae'] / mae_random) * 100
print(f"✓ Improvement over random: {best_vs_random:.1f}%")

print(f"\n📊 Results saved to: {OUTPUT_DIR}/")
print("=" * 100)
