# 💻 Scripts Directory

This directory contains all Python scripts for the news-stock prediction analysis pipeline.

## 📁 Directory Structure

```
02-scripts/
├── Core Analysis Pipeline
│   ├── 00_data_acquisition.py        # Data collection from APIs
│   ├── 01_data_loader.py             # Data loading and preprocessing
│   ├── 02_beta_estimation.py         # Factor model estimation
│   ├── 03_abnormal_returns.py        # Abnormal returns calculation
│   ├── 04_statistical_tests.py       # Hypothesis testing
│   └── 05_main_analysis.py           # Complete analysis pipeline
│
├── News Processing
│   ├── create_filtered_news.py       # Multi-stage news filtering
│   ├── create_stricter_filter.py     # Additional filtering strategies
│   ├── explore_news_data.py          # News data exploration
│   └── fix_event_dates.py            # Event date alignment
│
├── Analysis Utilities
│   ├── run_filtered_event_study.py   # Run specific event study
│   ├── run_final_event_study.py      # Final analysis run
│   └── debug_analysis.py             # Debugging utilities
│
├── Testing & Validation
│   ├── test_finnhub_news_availability.py
│   ├── test_finnhub_extended.py
│   └── test_marketaux_2024.py
│
├── Visualization & Reporting
│   ├── create_simple_presentation.py    # Generate presentation materials
│   └── create_detailed_presentation.py  # Detailed visualizations
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Run Complete Analysis

```bash
cd 02-scripts
python 05_main_analysis.py
```

This runs the entire pipeline:
1. Loads data from `01-data/`
2. Estimates factor models
3. Calculates abnormal returns
4. Runs statistical tests
5. Generates visualizations
6. Saves results to `03-output/`

### Generate Presentation Materials

```bash
python create_simple_presentation.py
```

Creates publication-quality visualizations and comprehensive document in `03-output/presentation/`.

## 📜 Script Descriptions

### Core Analysis Pipeline

#### `00_data_acquisition.py`
**Purpose**: Collect stock prices, news, and factor data from APIs

**Usage**:
```bash
python 00_data_acquisition.py
```

**What it does**:
- Downloads stock data from Yahoo Finance (yfinance)
- Collects news from EODHD API
- Downloads Fama-French factors
- Saves raw data to `01-data/`

**Requirements**:
- Internet connection
- API keys in `.env` file (for news)
- ~10-15 minutes runtime

**Output**:
- `AAPL_stock_data.csv`
- `TSLA_stock_data.csv`
- `AAPL_eodhd_news.csv` (large file)
- `TSLA_eodhd_news.csv` (large file)
- `fama_french_factors.csv`

---

#### `01_data_loader.py`
**Purpose**: Load and preprocess data for analysis

**Key Classes**:
```python
class DataLoader:
    def load_stock_data(ticker: str) -> pd.DataFrame
    def load_news_dates(ticker: str) -> pd.DataFrame
    def load_fama_french_factors() -> pd.DataFrame
    def merge_all_data() -> pd.DataFrame
```

**Usage**:
```python
from data_loader import DataLoader

loader = DataLoader(data_dir='../01-data')
stock_data = loader.load_stock_data('AAPL')
news_dates = loader.load_news_dates('AAPL')
ff_factors = loader.load_fama_french_factors()
```

**Features**:
- Handles missing data
- Aligns dates across datasets
- Calculates log returns
- Merges news dates with trading days

---

#### `02_beta_estimation.py`
**Purpose**: Estimate Fama-French factor loadings (betas)

**Key Classes**:
```python
class BetaEstimator:
    def estimate_betas(returns, factors) -> Dict[str, float]
    def rolling_estimation(window=126) -> pd.DataFrame
    def test_beta_stability() -> pd.DataFrame
```

**Methodology**:
- **Model**: R_i - R_f = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε
- **Estimation**: OLS regression
- **Window**: Full sample (for stability)
- **Output**: Beta estimates, standard errors, t-statistics, R²

**Usage**:
```python
from beta_estimation import BetaEstimator

estimator = BetaEstimator()
betas = estimator.estimate_betas(excess_returns, ff_factors)
print(f"Market Beta: {betas['Mkt-RF']:.3f}")
print(f"R²: {betas['R_squared']:.3f}")
```

**Output**:
- `beta_estimates.csv`: Beta coefficients for each factor
- `beta_stability.csv`: Rolling window estimates (if applicable)

---

#### `03_abnormal_returns.py`
**Purpose**: Calculate abnormal returns using estimated betas

**Key Classes**:
```python
class AbnormalReturnsCalculator:
    def calculate_expected_returns(betas, factors) -> pd.Series
    def calculate_abnormal_returns(actual, expected) -> pd.Series
    def calculate_cumulative_ar() -> pd.DataFrame
```

**Formulas**:
```
Expected Return = β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA)
Abnormal Return = Actual Return - Expected Return
CAR = Σ AR (cumulative sum)
```

**Usage**:
```python
from abnormal_returns import AbnormalReturnsCalculator

calc = AbnormalReturnsCalculator(betas=betas)
expected_returns = calc.calculate_expected_returns(ff_factors)
abnormal_returns = calc.calculate_abnormal_returns(actual_returns, expected_returns)
```

**Output**:
- `abnormal_returns.csv`: Daily abnormal returns with news day indicators
- `cumulative_abnormal_returns.csv`: Cumulative AR over time
- `ar_statistics.csv`: Summary statistics by group (news/non-news)

---

#### `04_statistical_tests.py`
**Purpose**: Conduct hypothesis tests on abnormal returns

**Key Classes**:
```python
class StatisticalTester:
    def one_sample_ttest(ar, H0=0) -> Dict
    def welchs_ttest(ar_news, ar_non_news) -> Dict
    def f_test_variance(ar_news, ar_non_news) -> Dict
    def ols_regression(ar, news_dummy) -> Dict
    def run_all_tests() -> pd.DataFrame
```

**Tests Conducted**:
1. **One-Sample t-test (Event Days)**: H₀: μ = 0
2. **One-Sample t-test (Non-Event Days)**: H₀: μ = 0
3. **Welch's t-test**: H₀: μ_news = μ_non_news
4. **F-test**: H₀: σ²_news = σ²_non_news
5. **OLS Regression**: AR = β₀ + β₁(News_Dummy) + ε

**Usage**:
```python
from statistical_tests import StatisticalTester

tester = StatisticalTester(significance_level=0.05)
results = tester.run_all_tests(abnormal_returns, news_days)
print(results['statistical_tests'])
```

**Output**:
- `statistical_tests.csv`: All test results (t-stats, p-values, significance)

---

#### `05_main_analysis.py`
**Purpose**: Orchestrate complete analysis pipeline

**What it does**:
1. Loads all data using `DataLoader`
2. Estimates betas using `BetaEstimator`
3. Calculates ARs using `AbnormalReturnsCalculator`
4. Runs tests using `StatisticalTester`
5. Generates visualizations
6. Saves all results

**Usage**:
```bash
python 05_main_analysis.py
```

**Runtime**: ~2-5 minutes (depending on data size)

**Output**: Complete analysis results in `03-output/TICKER_improved_study/`

---

### News Processing Scripts

#### `create_filtered_news.py`
**Purpose**: Filter raw news through multi-stage pipeline

**Filtering Stages**:
1. Sentiment filtering (|polarity| > 0.5)
2. Category filtering (priority categories only)
3. Volume filtering (top 10% days)
4. One-per-day selection
5. Trading day alignment

**Usage**:
```bash
python create_filtered_news.py
```

**Output**:
- `AAPL_improved_events.csv`
- `AAPL_improved_event_dates.csv`
- `TSLA_improved_events.csv`
- `TSLA_improved_event_dates.csv`

---

### Visualization Scripts

#### `create_simple_presentation.py`
**Purpose**: Generate publication-quality visualizations and comprehensive report

**What it creates**:
1. **Overview Comparison**: AAPL vs TSLA side-by-side (6 metrics)
2. **Detailed Analysis**: AR distributions, time series, box plots, factor loadings
3. **News Characteristics**: Sentiment distribution, content length, coverage
4. **Comprehensive Document**: 50+ page detailed report (Markdown)

**Usage**:
```bash
python create_simple_presentation.py
```

**Output Location**: `../03-output/presentation/`

**Files Created**:
- `overview_comparison.png` (300 DPI)
- `AAPL_detailed_analysis.png` (300 DPI)
- `TSLA_detailed_analysis.png` (300 DPI)
- `news_characteristics.png` (300 DPI)
- `DETAILED_PRESENTATION_DOCUMENT.md`

**Runtime**: ~30-60 seconds

---

## 🎯 Common Workflows

### Workflow 1: First-Time Setup and Analysis

```bash
# 1. Collect data (if not already present)
python 00_data_acquisition.py

# 2. Filter news events
python create_filtered_news.py

# 3. Run complete analysis
python 05_main_analysis.py

# 4. Generate presentation materials
python create_simple_presentation.py
```

### Workflow 2: Re-run Analysis with Existing Data

```bash
# Skip data collection, use existing files
python 05_main_analysis.py
python create_simple_presentation.py
```

### Workflow 3: Custom Analysis

```python
# Custom script example
from data_loader import DataLoader
from beta_estimation import BetaEstimator
from abnormal_returns import AbnormalReturnsCalculator
from statistical_tests import StatisticalTester

# Load data
loader = DataLoader()
data = loader.load_all_data('AAPL')

# Estimate betas
estimator = BetaEstimator()
betas = estimator.estimate_betas(data)

# Calculate ARs
calc = AbnormalReturnsCalculator(betas)
ar = calc.calculate_abnormal_returns(data)

# Run custom tests
tester = StatisticalTester()
results = tester.one_sample_ttest(ar[ar['News_Day']==1]['AR'])
print(f"P-value: {results['p_value']:.4f}")
```

---

## 🧪 Testing Scripts

### `test_finnhub_extended.py`
Tests how far back Finnhub news data goes (data availability check).

```bash
python test_finnhub_extended.py
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Required for data collection
FINNHUB_API_KEY=your_finnhub_key_here
EODHD_API_KEY=your_eodhd_key_here
ALPHA_VANTAGE_API_KEY=your_alphavantage_key_here (optional)
```

### Script Parameters

Most scripts accept command-line arguments:

```bash
# Run analysis for specific ticker
python 05_main_analysis.py --ticker AAPL

# Use custom data directory
python 05_main_analysis.py --data-dir /path/to/data

# Set output directory
python 05_main_analysis.py --output-dir /path/to/output

# Specify event window
python 05_main_analysis.py --event-window -1,2

# Set significance level
python 05_main_analysis.py --alpha 0.01
```

---

## 🐛 Debugging

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
**Solution**:
```bash
pip install -r requirements.txt
```

**Issue**: `KeyError: 'Date'` in analysis
**Solution**: Check that data files have correct column names. Run:
```bash
python -c "import pandas as pd; print(pd.read_csv('../01-data/AAPL_stock_data.csv').columns)"
```

**Issue**: `FileNotFoundError: ../01-data/AAPL_improved_event_dates.csv`
**Solution**: Run news filtering first:
```bash
python create_filtered_news.py
```

**Issue**: API rate limits exceeded
**Solution**:
- Wait for rate limit reset
- Use free tier sparingly
- Consider paid API tier

### Debug Mode

Most scripts support verbose output:
```bash
python 05_main_analysis.py --verbose
python 05_main_analysis.py --debug
```

---

## 📊 Performance Optimization

### Runtime Optimization

**Data Loading**: Use `pandas` optimizations
```python
# Read only needed columns
df = pd.read_csv('data.csv', usecols=['Date', 'Close'])

# Use dtype specification
df = pd.read_csv('data.csv', dtype={'Close': 'float32'})

# Read in chunks for large files
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

**Parallel Processing**: For multiple tickers
```python
from multiprocessing import Pool

def analyze_ticker(ticker):
    # Run analysis
    pass

with Pool(4) as pool:
    results = pool.map(analyze_ticker, ['AAPL', 'TSLA', 'GOOGL', 'AMZN'])
```

---

## 📚 Code Style & Standards

### Style Guide
- **PEP 8**: Python style guide
- **Type Hints**: Use for function signatures
- **Docstrings**: Google-style docstrings
- **Black**: Code formatter

### Example:
```python
def calculate_abnormal_returns(
    actual_returns: pd.Series,
    expected_returns: pd.Series,
    news_days: pd.Series
) -> pd.DataFrame:
    """
    Calculate abnormal returns for each day.

    Args:
        actual_returns: Series of actual stock returns
        expected_returns: Series of expected returns from factor model
        news_days: Boolean series indicating news days

    Returns:
        DataFrame with abnormal returns and statistics

    Example:
        >>> ar_df = calculate_abnormal_returns(returns, expected, news)
        >>> print(ar_df.head())
    """
    ar = actual_returns - expected_returns
    return pd.DataFrame({
        'Date': actual_returns.index,
        'AR': ar,
        'News_Day': news_days
    })
```

---

## 🤝 Contributing

To add a new script:

1. Follow naming convention: `##_description.py` (## = sequence number)
2. Add comprehensive docstring
3. Include command-line interface if applicable
4. Add to this README under appropriate section
5. Write unit tests (in `tests/` directory)
6. Update main README if significant feature

---

**Last Updated**: October 11, 2024
