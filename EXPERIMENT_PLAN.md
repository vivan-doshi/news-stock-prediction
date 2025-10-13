# News-Stock Prediction Experiment Plan

## 📊 Current Stock Configuration

### Summary Statistics
- **Total Stocks**: 54 (no duplicates)
- **Total Sectors**: 10
- **Stocks per Sector**: Minimum 5, Maximum 7
- **Sector ETFs**: 10 (all downloaded ✓)

### Sector Breakdown (All have ≥5 stocks)

| Sector | ETF | Count | Stocks |
|--------|-----|-------|--------|
| **Technology** | XLK | 7 | AAPL, ADBE, AVGO, CRM, MSFT, NVDA, ORCL |
| **Finance** | XLF | 6 | BAC, BLK, GS, JPM, MS, WFC |
| **Healthcare** | XLV | 6 | ABBV, JNJ, LLY, PFE, TMO, UNH |
| **Communication Services** | XLC | 5 | CMCSA, DIS, GOOGL, META, NFLX |
| **Consumer Discretionary** | XLY | 5 | AMZN, HD, MCD, NKE, TSLA |
| **Consumer Staples** | XLP | 5 | COST, KO, PEP, PG, WMT |
| **Industrials** | XLI | 5 | BA, CAT, GE, HON, UNP |
| **Energy** | XLE | 5 | COP, CVX, EOG, SLB, XOM |
| **Real Estate** | XLRE | 5 | AMT, CCI, EQIX, PLD, SPG |
| **Utilities** | XLU | 5 | AEP, D, DUK, NEE, SO |

---

## 📁 Data Availability Status

### Stock Price Data
- ✓ **Available**: 17/54 stocks (31%)
  - AAPL, AMZN, AMT, BA, GOOGL, GS, JNJ, JPM, META, MSFT, NEE, NVDA, PFE, PG, TSLA, WMT, XOM

- ✗ **Missing**: 37/54 stocks (69%)
  - ABBV, ADBE, AEP, AVGO, BAC, BLK, CAT, CCI, CMCSA, COP, COST, CRM, CVX, D, DIS, DUK, EOG, EQIX, GE, HD, HON, KO, LLY, MCD, MS, NFLX, NKE, ORCL, PEP, PLD, SLB, SO, SPG, TMO, UNH, UNP, WFC

### News Data
- ✓ **Available**: 17/54 stocks (31%)
  - Same as stock data (matched)

- ✗ **Missing**: 37/54 stocks (69%)
  - Same as stock data (matched)

### ETF Data
- ✓ **Available**: 10/10 ETFs (100%) ✓
  - All sector ETFs downloaded

### Fama-French Factor Data
- ✓ Available: 5-factor model data

---

## 🎯 Next Steps to Fix Experimentation

### Phase 1: Data Collection (Priority: HIGH)
**Goal**: Download missing stock and news data for 37 stocks

#### 1.1 Download Missing Stock Data
```python
# Create script: 17_download_missing_stocks.py
missing_stocks = [
    'ABBV', 'ADBE', 'AEP', 'AVGO', 'BAC', 'BLK', 'CAT', 'CCI',
    'CMCSA', 'COP', 'COST', 'CRM', 'CVX', 'D', 'DIS', 'DUK',
    'EOG', 'EQIX', 'GE', 'HD', 'HON', 'KO', 'LLY', 'MCD',
    'MS', 'NFLX', 'NKE', 'ORCL', 'PEP', 'PLD', 'SLB', 'SO',
    'SPG', 'TMO', 'UNH', 'UNP', 'WFC'
]
```

**Actions**:
- [ ] Create download script using EODHD API
- [ ] Download 2022-2024 stock price data (same as existing)
- [ ] Verify data quality (no gaps, proper OHLCV format)
- [ ] Estimated time: ~10 minutes (API rate limits)

#### 1.2 Download Missing News Data
```python
# Create script: 18_download_missing_news.py
```

**Actions**:
- [ ] Use EODHD News API for all 37 missing stocks
- [ ] Date range: 2022-01-01 to 2024-12-31
- [ ] Save in same format as existing news data
- [ ] Estimated time: ~15-20 minutes (larger dataset)

---

### Phase 2: Data Validation (Priority: HIGH)
**Goal**: Ensure data quality and consistency

#### 2.1 Quality Checks
- [ ] Check for missing dates in stock data
- [ ] Verify news article counts per stock (should be reasonable)
- [ ] Validate date ranges match (2022-2024)
- [ ] Check for outliers in returns
- [ ] Ensure ETF data aligns with stock dates

#### 2.2 Create Data Summary Report
```python
# Create script: 19_data_summary_report.py
```

**Outputs**:
- Stock data coverage by date
- News article counts by stock/sector
- Missing data gaps identification
- Data quality metrics

---

### Phase 3: Feature Engineering (Priority: MEDIUM)
**Goal**: Prepare consistent features across all stocks

#### 3.1 Stock Return Features
- [ ] Calculate daily returns for all 54 stocks
- [ ] Calculate sector-relative returns (stock vs ETF)
- [ ] Calculate market-relative returns (stock vs SP500)
- [ ] Create volatility features (rolling std)

#### 3.2 News Features
- [ ] Sentiment analysis (if not already done)
- [ ] News volume features (count per day/week)
- [ ] News timing features (days since last news)
- [ ] Aggregate at multiple time windows (1d, 3d, 7d, 30d)

#### 3.3 Fama-French Integration
- [ ] Merge FF 5-factor data with all stocks
- [ ] Calculate factor loadings per stock
- [ ] Create factor-adjusted returns

---

### Phase 4: Experimental Design (Priority: HIGH)
**Goal**: Set up robust experiments with proper controls

#### 4.1 Train/Test Split Strategy
```python
# Chronological split
train_period = "2022-01-01" to "2023-12-31"  # 2 years
test_period  = "2024-01-01" to "2024-12-31"  # 1 year
```

#### 4.2 Cross-Validation Strategy
```python
# Time-series cross-validation
# Window-based approach for temporal data
```

#### 4.3 Baseline Models
- [ ] Random baseline (random predictions)
- [ ] Momentum baseline (past returns predict future)
- [ ] Mean reversion baseline
- [ ] Market model (beta-based prediction)

#### 4.4 Experimental Models
- [ ] News-only model (news sentiment → returns)
- [ ] Price-only model (technical indicators → returns)
- [ ] News + Price combined model
- [ ] Sector-aware models
- [ ] Factor-adjusted models

---

### Phase 5: Sector-Level Analysis (Priority: MEDIUM)
**Goal**: Leverage multiple stocks per sector for robust insights

#### 5.1 Within-Sector Analysis
```python
# For each sector (e.g., Technology with 7 stocks):
# 1. Does news predict relative performance within sector?
# 2. Do sector-wide news events affect all stocks similarly?
# 3. Stock-specific vs sector-wide news separation
```

#### 5.2 Cross-Sector Comparison
- [ ] Which sectors show strongest news-return relationship?
- [ ] News sentiment effectiveness by sector
- [ ] Volatility patterns by sector
- [ ] Event study analysis by sector

#### 5.3 Portfolio-Level Testing
- [ ] Equal-weighted sector portfolios
- [ ] News-weighted portfolios
- [ ] Risk-adjusted performance metrics

---

### Phase 6: Model Evaluation Framework (Priority: HIGH)
**Goal**: Comprehensive evaluation metrics

#### 6.1 Prediction Metrics
- [ ] Direction accuracy (up/down classification)
- [ ] Magnitude metrics (MAE, RMSE, R²)
- [ ] Sharpe ratio (risk-adjusted returns)
- [ ] Information ratio (vs benchmark)

#### 6.2 Statistical Tests
- [ ] Significance testing (are results better than random?)
- [ ] Out-of-sample validation
- [ ] Robustness checks (different time periods)

#### 6.3 Economic Metrics
- [ ] Trading strategy P&L
- [ ] Transaction cost analysis
- [ ] Maximum drawdown
- [ ] Win rate and profit factor

---

## 🔧 Technical Implementation Plan

### Scripts to Create

1. **17_download_missing_stocks.py**
   - Download 37 missing stock price datasets
   - Use existing EODHD infrastructure

2. **18_download_missing_news.py**
   - Download news for 37 missing stocks
   - Match date ranges with existing data

3. **19_data_summary_report.py**
   - Generate comprehensive data quality report
   - Identify any remaining gaps

4. **20_unified_feature_engineering.py**
   - Process all 54 stocks consistently
   - Create standardized feature set

5. **21_sector_analysis.py**
   - Sector-level aggregations
   - Cross-sector comparisons

6. **22_baseline_models.py**
   - Implement all baseline models
   - Establish performance benchmarks

7. **23_experimental_models.py**
   - News-based prediction models
   - Combined models

8. **24_evaluation_framework.py**
   - Comprehensive evaluation
   - Statistical testing
   - Results visualization

---

## 📈 Expected Outcomes

### Short-term (After Phase 1-2)
- Complete dataset of 54 stocks × ~750 days = 40,500 observations
- News data for all stocks
- Data quality report

### Medium-term (After Phase 3-4)
- Baseline model performance benchmarks
- Initial news prediction models
- Sector-specific insights

### Long-term (After Phase 5-6)
- Publication-ready results
- Robust experimental findings
- Cross-sector comparisons
- Trading strategy validation

---

## 🚨 Potential Challenges & Solutions

### Challenge 1: API Rate Limits
**Solution**: Implement batching with delays, use multiple API keys if available

### Challenge 2: News Data Quality
**Solution**: Implement robust filtering, deduplication, and quality checks

### Challenge 3: Sector Imbalance
**Solution**: Use sector-adjusted metrics, equal-weighted portfolios

### Challenge 4: Temporal Dependencies
**Solution**: Use proper time-series CV, avoid look-ahead bias

### Challenge 5: Multiple Testing
**Solution**: Bonferroni correction, FDR control, pre-registered hypotheses

---

## 📋 Immediate Action Items

### Priority 1 (Do First)
1. ✓ Create expanded stock configuration (DONE)
2. Run `17_download_missing_stocks.py` to get stock data
3. Run `18_download_missing_news.py` to get news data
4. Run `19_data_summary_report.py` to validate

### Priority 2 (Do Next)
5. Create unified feature engineering pipeline
6. Implement baseline models
7. Set up evaluation framework

### Priority 3 (Do After)
8. Run sector-level analysis
9. Implement experimental models
10. Generate comprehensive results report

---

## 🎓 Research Questions to Answer

### Primary Questions
1. **Does news predict stock returns better than chance?**
   - Hypothesis: Yes, with statistical significance
   - Test: Compare news-based models vs random baseline

2. **Does news work better for certain sectors?**
   - Hypothesis: Higher information asymmetry sectors (tech, healthcare) show stronger effects
   - Test: Sector-stratified analysis

3. **What is the optimal news aggregation window?**
   - Hypothesis: 3-7 day windows work best
   - Test: Grid search over windows (1d, 3d, 7d, 14d, 30d)

### Secondary Questions
4. Do sector-wide news events predict sector returns?
5. Is stock-specific news more predictive than general news?
6. Does combining news with price data improve predictions?
7. Are news effects stronger during high volatility periods?

---

## ✅ Success Criteria

### Data Collection Success
- [ ] All 54 stocks have complete price data (2022-2024)
- [ ] All 54 stocks have news data with >100 articles each
- [ ] No critical data gaps (>5 consecutive missing days)

### Model Success
- [ ] News model significantly outperforms random baseline (p < 0.05)
- [ ] Direction accuracy >50% on out-of-sample data
- [ ] At least 1 sector shows strong news-return relationship

### Research Success
- [ ] Answer all primary research questions with statistical evidence
- [ ] Identify actionable insights for trading/investment
- [ ] Produce visualizations and summary report

---

**Last Updated**: 2025-10-12
**Status**: Ready to begin Phase 1 data collection
