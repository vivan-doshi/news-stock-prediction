# News-Stock Prediction

Analyzing and predicting the impact of news events on stock prices using event study methodology and machine learning.

## Project Overview

This project aims to build an automated system that predicts both the direction and magnitude of stock price movements following news events.

## Project Phases

### Phase 1: News Impact Detection (Current Phase)
**Objective:** Detect if news significantly moves stock prices using event study methodology

**Methodology:**
- **Expected Returns Model:** Fama-French 5-factor model + sector factor
- **Beta Estimation:** Rolling 126-day window using OLS regression, excluding major news days
- **Abnormal Returns (AR):** Calculated as Actual Return - Expected Return
- **Event Window:** [-1, +2] days around news events
- **Statistical Testing:**
  - One-sample t-test (H0: AR = 0)
  - Two-sample t-test (news vs non-news days)
  - Variance tests (F-test)
  - Regression analysis

**Success Criteria:**
- Significant t-test results (p < 0.05)
- Mean abnormal return of 1-3% on news days
- Robust results across different time periods and specifications
- AR near 0% on non-news days (good model fit)

**Key Steps:**
1. Data collection (stock prices, news dates, Fama-French factors)
2. Beta estimation with rolling windows
3. Expected return calculation
4. Abnormal return calculation
5. Statistical significance testing
6. Robustness checks (subperiod analysis, outlier sensitivity)
7. Visualization and reporting

### Phase 2: Sentiment Classification (Planned)
**Objective:** Classify news sentiment as positive or negative

**Approach:**
- NLP analysis using FinBERT or GPT-4
- Training/test split for model validation
- Requires full news text data

### Phase 3: Magnitude Prediction (Planned)
**Objective:** Predict the magnitude of price impact

**Approach:**
- Regression modeling with features from news and market data
- Combine sentiment classification with quantitative predictions
- Build automated prediction system

## Technical Stack
- Python for data analysis and modeling
- Statistical methods: OLS regression, event study methodology
- Machine learning for sentiment analysis and prediction
