# Category-Specific Event Study Report

**Analysis Date:** 2025-10-13 03:20:41

**Methodology:** Fama-French 5-Factor Model with Category-Specific News Events

---

## Executive Summary

This report presents a comprehensive analysis of **how different types of news** impact stock returns across **50 stocks** in **10 sectors**.

### Key Findings

**Overall Statistics:**
- **Total Category-Stock Combinations:** 400
- **Total News Events Analyzed:** 139,342
- **Significant Results:** 147 / 400 (36.8%)

---

## Category Effectiveness Rankings

### Top 5 Most Impactful News Categories

| Rank | Category | Avg Effect Size (Cohen's d) | % Stocks Significant | Total Events | Effectiveness Score |
|------|----------|------------------------------|---------------------|--------------|---------------------|
| 1 | Analyst_Ratings | 0.095 | 44.0% | 23081 | 0.042 |
| 2 | Earnings | 0.086 | 48.0% | 23034 | 0.041 |
| 3 | Dividends | 0.075 | 36.0% | 10596 | 0.027 |
| 4 | Market_Performance | 0.052 | 46.0% | 26535 | 0.024 |
| 5 | Executive_Changes | 0.037 | 40.0% | 21122 | 0.015 |


### Category Performance Summary

| Category | Avg AR (Event) | Avg AR (Non-Event) | Difference | Avg p-value | Significant Stocks | Effect Size |
|----------|----------------|-------------------|------------|-------------|-------------------|-------------|
| Analyst_Ratings | 0.0124 | 0.0110 | 0.0014 | 0.2790 | 22/50 | 0.095 |
| Earnings | 0.0123 | 0.0110 | 0.0013 | 0.2511 | 24/50 | 0.086 |
| Dividends | 0.0126 | 0.0114 | 0.0012 | 0.2884 | 18/50 | 0.075 |
| Market_Performance | 0.0120 | 0.0112 | 0.0008 | 0.1959 | 23/50 | 0.052 |
| Executive_Changes | 0.0119 | 0.0113 | 0.0006 | 0.3037 | 20/50 | 0.037 |
| Product_Launch | 0.0120 | 0.0114 | 0.0006 | 0.3147 | 15/50 | 0.038 |
| M&A | 0.0120 | 0.0115 | 0.0005 | 0.3508 | 13/50 | 0.038 |
| Regulatory_Legal | 0.0118 | 0.0115 | 0.0003 | 0.3642 | 12/50 | 0.016 |


---

## Sector-Level Analysis

### Category Impact by Sector


#### Communication Services

**Most Impactful Categories:**
1. **Analyst_Ratings**: positive impact (d=0.169), 2/5 stocks significant
1. **Earnings**: positive impact (d=0.161), 2/5 stocks significant
1. **Executive_Changes**: positive impact (d=0.150), 2/5 stocks significant


#### Consumer Discretionary

**Most Impactful Categories:**
1. **Analyst_Ratings**: positive impact (d=0.094), 2/5 stocks significant
1. **Earnings**: positive impact (d=0.090), 2/5 stocks significant
1. **Dividends**: positive impact (d=0.087), 1/5 stocks significant


#### Consumer Staples

**Most Impactful Categories:**
1. **Analyst_Ratings**: positive impact (d=0.201), 3/5 stocks significant
1. **Earnings**: positive impact (d=0.181), 3/5 stocks significant
1. **Market_Performance**: positive impact (d=0.176), 2/5 stocks significant


#### Energy

**Most Impactful Categories:**
1. **Dividends**: negative impact (d=-0.012), 1/5 stocks significant
1. **M&A**: negative impact (d=-0.017), 2/5 stocks significant
1. **Analyst_Ratings**: negative impact (d=-0.038), 2/5 stocks significant


#### Finance

**Most Impactful Categories:**
1. **Dividends**: positive impact (d=0.120), 3/5 stocks significant
1. **Analyst_Ratings**: positive impact (d=0.088), 2/5 stocks significant
1. **Earnings**: positive impact (d=0.085), 2/5 stocks significant


#### Healthcare

**Most Impactful Categories:**
1. **Analyst_Ratings**: positive impact (d=0.148), 3/5 stocks significant
1. **Earnings**: positive impact (d=0.127), 3/5 stocks significant
1. **M&A**: positive impact (d=0.122), 4/5 stocks significant


#### Industrials

**Most Impactful Categories:**
1. **Analyst_Ratings**: positive impact (d=0.095), 2/5 stocks significant
1. **Earnings**: positive impact (d=0.064), 2/5 stocks significant
1. **M&A**: positive impact (d=0.032), 0/5 stocks significant


#### Real Estate

**Most Impactful Categories:**
1. **Dividends**: positive impact (d=0.103), 2/5 stocks significant
1. **Regulatory_Legal**: positive impact (d=0.093), 0/5 stocks significant
1. **Product_Launch**: positive impact (d=0.058), 1/5 stocks significant


#### Technology

**Most Impactful Categories:**
1. **Earnings**: positive impact (d=0.141), 4/5 stocks significant
1. **Analyst_Ratings**: positive impact (d=0.137), 4/5 stocks significant
1. **Executive_Changes**: positive impact (d=0.131), 4/5 stocks significant


#### Utilities

**Most Impactful Categories:**
1. **Regulatory_Legal**: positive impact (d=0.022), 1/5 stocks significant
1. **Product_Launch**: positive impact (d=0.020), 1/5 stocks significant
1. **Earnings**: positive impact (d=0.014), 2/5 stocks significant


---

## Key Insights

### 1. Category Effectiveness

**Most Effective Categories:**
['Analyst_Ratings', 'Earnings', 'Dividends']

These categories show the strongest and most consistent impact on stock returns across sectors.

### 2. Sector Sensitivity

Different sectors show varying sensitivity to news categories:

- **Consumer Staples + Analyst_Ratings**: d=0.201, 60% significant
- **Consumer Staples + Earnings**: d=0.181, 60% significant
- **Consumer Staples + Market_Performance**: d=0.176, 40% significant
- **Communication Services + Analyst_Ratings**: d=0.169, 40% significant
- **Communication Services + Earnings**: d=0.161, 40% significant


### 3. Statistical Robustness

- Average model fit (R²): 0.386
- Median p-value: 0.1675
- Overall significance rate: 36.8%

---

## Methodology

### News Categories Analyzed

1. **Earnings**: Quarterly/annual earnings reports, guidance updates
2. **Product Launch**: New product announcements, major releases
3. **Executive Changes**: CEO changes, board appointments, executive departures
4. **M&A**: Mergers, acquisitions, strategic partnerships
5. **Regulatory/Legal**: Regulatory approvals/denials, lawsuits, compliance issues
6. **Analyst Ratings**: Analyst upgrades/downgrades, price target changes
7. **Dividends**: Dividend announcements, changes, special dividends
8. **Market Performance**: Milestone achievements, market share changes

### Statistical Methods

1. **Factor Model:** Fama-French 5-Factor (Mkt-RF, SMB, HML, RMW, CMA)
2. **Rolling Window:** 252-day estimation window
3. **Abnormal Returns:** Winsorized at 1% tails
4. **Hypothesis Tests:** Welch's t-test, Mann-Whitney U test
5. **Effect Size:** Cohen's d
6. **Confidence Intervals:** Bootstrap (1000 iterations, 95% CI)

---

## Visualizations

The following visualizations are available:

1. **`sector_category_heatmap.png`** - Overall impact across sectors and categories
2. **`category_effectiveness_ranking.png`** - Category performance comparison
3. **`by_sector/[SECTOR]/category_comparison.png`** - Sector-specific analysis (10 files)
4. **`results/[TICKER]/[CATEGORY]/event_study.png`** - Individual analyses (400 files)

---

## Data Files

### Summary Files
- `comprehensive_category_results.csv` - All 400 category-stock results
- `category_summary_statistics.csv` - Aggregated category statistics
- `category_sector_matrix.csv` - Sector × category event counts

### Sector Files
- `by_sector/[SECTOR]/category_summary.csv` - Category statistics per sector (10 files)

### Individual Stock-Category Files
- `results/[TICKER]/[CATEGORY]/summary.csv` - Detailed results (400 files)

---

## Recommendations for Next Phase

Based on this analysis, we recommend:

1. **Focus on High-Impact Categories**: Analyst_Ratings, Earnings, Dividends
   - These show strongest and most consistent effects

2. **Sector-Specific Strategies**:
   - **Communication Services**: Prioritize Analyst_Ratings news (d=0.169)
   - **Consumer Discretionary**: Prioritize Analyst_Ratings news (d=0.094)
   - **Consumer Staples**: Prioritize Analyst_Ratings news (d=0.201)
   - **Energy**: Prioritize Market_Performance news (d=-0.088)
   - **Finance**: Prioritize Dividends news (d=0.120)
   - **Healthcare**: Prioritize M&A news (d=0.122)
   - **Industrials**: Prioritize Analyst_Ratings news (d=0.095)
   - **Real Estate**: Prioritize Dividends news (d=0.103)
   - **Technology**: Prioritize Earnings news (d=0.141)
   - **Utilities**: Prioritize Executive_Changes news (d=-0.064)


3. **Model Refinement**:
   - Consider category-specific event windows (some categories may have delayed/prolonged effects)
   - Investigate sentiment interaction (positive vs negative news within categories)
   - Explore category combinations (multiple simultaneous events)

---

## References

- Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature*, 35(1), 13-39.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.

---

*Report generated by Category Event Study System*
*Analysis Date: 2025-10-13 03:20:41*
