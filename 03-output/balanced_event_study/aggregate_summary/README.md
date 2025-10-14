# Event Study Analysis - Complete Summary Report
## All Stocks Across Multiple Sectors (2021-2025)

---

## 📊 Executive Summary

This comprehensive event study analysis examines the impact of news events on stock returns for **41 stocks** across **9 sectors** over the period **2021-2025**.

### Key Findings:

- **21 out of 41 stocks (51.2%)** showed **statistically significant** news impact (p < 0.05)
- **15 stocks** had **positive news impact** (higher returns on news days)
- **6 stocks** had **negative news impact** (higher returns on non-news days)
- **20 stocks** showed no significant difference between news and non-news days

---

## 📈 Overall Statistics

| Metric | Value |
|--------|-------|
| **Total Stocks Analyzed** | 41 |
| **Significant Results** | 21 (51.2%) |
| **Non-Significant Results** | 20 (48.8%) |
| **Average Difference in AR** | 0.074% |
| **Median Difference in AR** | 0.103% |
| **Average \|Cohen's d\|** | 0.158 |

---

## 🏆 Top Performers

### Top 10 Strongest POSITIVE News Impacts
*(Higher returns on news days)*

| Rank | Ticker | Sector | Difference | P-Value | Cohen's d | Interpretation |
|------|--------|--------|------------|---------|-----------|----------------|
| 1 | **META** | Communication | **+0.967%** | <0.0001 | 0.563 | Large effect |
| 2 | **NVDA** | Technology | **+0.440%** | <0.0001 | 0.220 | Small effect |
| 3 | **LLY** | Healthcare | **+0.399%** | 0.0035 | 0.233 | Small effect |
| 4 | **AMT** | Real Estate | **+0.396%** | <0.0001 | 0.243 | Small effect |
| 5 | **WMT** | Consumer Staples | **+0.386%** | <0.0001 | 0.298 | Small effect |
| 6 | **NEE** | Energy | **+0.374%** | <0.0001 | 0.236 | Small effect |
| 7 | **PG** | Consumer Staples | **+0.347%** | <0.0001 | 0.286 | Small effect |
| 8 | **MSFT** | Technology | **+0.332%** | <0.0001 | 0.293 | Small effect |
| 9 | **JPM** | Finance | **+0.330%** | <0.0001 | 0.272 | Small effect |
| 10 | **AMZN** | Consumer Disc. | **+0.258%** | 0.0009 | 0.177 | Small effect |

### Top 6 Strongest NEGATIVE News Impacts
*(Higher returns on non-news days)*

| Rank | Ticker | Sector | Difference | P-Value | Cohen's d | Interpretation |
|------|--------|--------|------------|---------|-----------|----------------|
| 1 | **COP** | Energy | **-0.465%** | 0.0002 | -0.272 | Small effect |
| 2 | **AAPL** | Technology | **-0.429%** | <0.0001 | -0.329 | Small effect |
| 3 | **WFC** | Finance | **-0.350%** | 0.0030 | -0.222 | Small effect |
| 4 | **CVX** | Energy | **-0.331%** | 0.0064 | -0.236 | Small effect |
| 5 | **BAC** | Finance | **-0.309%** | 0.0074 | -0.221 | Small effect |
| 6 | **HON** | Industrials | **-0.202%** | 0.0330 | -0.156 | Small effect |

---

## 🏢 Sector-Level Analysis

### Sector Summary (Ranked by Average Difference)

| Sector | Avg AR (News) | Avg AR (Non-News) | Avg Difference | Significant Stocks | Significance Rate |
|--------|---------------|-------------------|----------------|-------------------|-------------------|
| **Communication** | 1.250% | 1.030% | **+0.220%** | 2/4 | 50.0% |
| **Consumer Staples** | 1.240% | 1.060% | **+0.180%** | 2/5 | 40.0% |
| **Healthcare** | 1.190% | 1.060% | **+0.130%** | 3/5 | 60.0% |
| **Technology** | 1.080% | 0.950% | **+0.130%** | 3/5 | 60.0% |
| **Consumer Discretionary** | 1.170% | 1.110% | **+0.060%** | 1/5 | 20.0% |
| **Real Estate** | 1.250% | 1.200% | **+0.040%** | 1/5 | 20.0% |
| **Industrials** | 1.080% | 1.060% | **+0.020%** | 2/2 | 100.0% |
| **Finance** | 1.140% | 1.200% | **-0.050%** | 4/5 | 80.0% |
| **Energy** | 1.170% | 1.230% | **-0.070%** | 3/5 | 60.0% |

### Key Sector Insights:

1. **Communication Sector**: Strongest positive news impact (+0.22%)
   - META shows exceptional news sensitivity (+0.967%)

2. **Finance Sector**: Negative news impact overall (-0.05%)
   - News days tend to underperform non-news days
   - 80% of finance stocks show significant effects

3. **Energy Sector**: Negative news impact (-0.07%)
   - Both positive (NEE) and negative (COP, CVX) effects present

4. **Industrials**: 100% significance rate
   - Small sample (2 stocks: BA, HON), but both show significant effects

5. **Consumer Discretionary & Real Estate**: Low significance rates (20%)
   - News impact is less pronounced in these sectors

---

## 📑 Generated Files

### Summary Tables:
- **[full_summary.csv](full_summary.csv)** - Complete results for all 41 stocks
- **[significant_results.csv](significant_results.csv)** - 21 stocks with significant effects only
- **[sector_summary.csv](sector_summary.csv)** - Aggregated sector-level statistics

### Visualizations:

#### Cross-Stock Comparisons:
1. **[01_ar_comparison_all_stocks.png](01_ar_comparison_all_stocks.png)** - Bar chart comparing AR on news vs non-news days
2. **[02_effect_size_all_stocks.png](02_effect_size_all_stocks.png)** - Horizontal bar chart of effect sizes (difference in AR)
3. **[03_pvalue_distribution.png](03_pvalue_distribution.png)** - P-value distribution (log scale)
4. **[04_cohens_d_effect_size.png](04_cohens_d_effect_size.png)** - Cohen's d effect sizes with benchmarks

#### Sector-Level Analysis:
5. **[05_sector_ar_comparison.png](05_sector_ar_comparison.png)** - Average AR by sector (news vs non-news)
6. **[06_sector_significance_rate.png](06_sector_significance_rate.png)** - % of stocks with significant effects by sector
7. **[07_sector_effect_distribution.png](07_sector_effect_distribution.png)** - Box plots showing effect size distributions
8. **[08_effect_vs_coverage.png](08_effect_vs_coverage.png)** - Scatter plot: effect size vs news coverage %

### Text Reports:
- **[SUMMARY_REPORT.txt](SUMMARY_REPORT.txt)** - Detailed text summary with all statistics

---

## 📊 Individual Stock Results

Each stock has its own detailed analysis folder:
```
03-output/balanced_event_study/
├── AAPL/
│   ├── robust_event_study.png
│   └── summary.csv
├── MSFT/
│   ├── robust_event_study.png
│   └── summary.csv
├── ... (39 more stocks)
```

Each folder contains:
- **robust_event_study.png**: 4-panel visualization with:
  - Abnormal returns over time
  - Distribution comparisons (news vs non-news)
  - Statistical test results
  - Bootstrap confidence intervals

- **summary.csv**: Complete statistical results including:
  - Mean/median/std of AR for news and non-news days
  - T-test, Mann-Whitney U, and permutation test results
  - Cohen's d effect size
  - Bootstrap confidence intervals

---

## 🔍 Interpretation Guide

### Statistical Significance:
- **P-value < 0.05**: Statistically significant difference
- **P-value ≥ 0.05**: No significant difference

### Effect Size (Cohen's d):
- **|d| < 0.2**: Negligible to small effect
- **0.2 ≤ |d| < 0.5**: Small effect
- **0.5 ≤ |d| < 0.8**: Medium effect
- **|d| ≥ 0.8**: Large effect

### Sign Interpretation:
- **Positive (+)**: News days have higher abnormal returns
- **Negative (-)**: Non-news days have higher abnormal returns

---

## 🎯 Key Takeaways

1. **News Impact is Stock-Specific**
   - 51% of stocks show significant news effects
   - Effects vary widely in magnitude and direction

2. **META Shows Exceptional News Sensitivity**
   - +0.967% difference (Cohen's d = 0.563)
   - Only stock with "medium" effect size

3. **Sector Patterns Emerge**
   - Tech & Consumer Staples: Generally positive news impact
   - Finance & Energy: Generally negative news impact

4. **Small but Significant Effects**
   - Most significant stocks show "small" effect sizes (|d| < 0.5)
   - Still economically meaningful over time

5. **20 Stocks Show No Significant Impact**
   - News may be already priced in
   - Or news quality/filtering issues
   - Market efficiency for these stocks

---

## 📝 Methodology

- **Date Range**: 2021-01-01 to 2025-07-31
- **Model**: Fama-French 5-Factor + Momentum
- **Winsorization**: 1% each tail (to handle outliers)
- **Bootstrap**: 1,000 iterations for confidence intervals
- **Statistical Tests**:
  - T-test (parametric)
  - Mann-Whitney U (non-parametric)
  - Permutation test (robust)

**Significant if all three tests show p < 0.05**

---

## 📌 Notes

- 9 stocks from the original 50-stock list had no available data (UNP, LMT, RTX, LIN, APD, SHW, ECL, DD, VZ)
- Analysis includes only stocks with sufficient news coverage (>100 articles over the period)
- News filtering uses balanced approach to control for data quality

---

**Generated**: October 13, 2025
**Analysis Script**: `26_robust_event_study_50_stocks.py`
**Aggregation Script**: `30_aggregate_results_and_visualizations.py`
