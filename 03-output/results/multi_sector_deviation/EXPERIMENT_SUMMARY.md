# Multi-Sector News Deviation Analysis - Summary Report

## Executive Summary

This analysis examined whether **extreme sentiment news** (|polarity| > 0.95) causes measurable deviations from sector performance across **6 major stocks** from different sectors over a **5-year period** (2019-2024).

### Key Question
**Do stocks deviate more from their sector on days with extreme news compared to regular days?**

### Main Finding
**No consistent evidence of news-driven sector deviation across stocks.**
- Only **1 out of 6 stocks** showed statistically significant results
- Most stocks showed **negative or minimal deviation increase** on news days
- The one significant result (XOM) actually showed **lower deviation** on news days (-22%)

---

## Methodology

### Deviation Concept
Instead of just measuring abnormal returns, we measured **stock-specific deviation**:
```
Deviation = Stock Abnormal Return - Sector Abnormal Return
```

This isolates stock-specific news impact from sector-wide movements.

### Analysis Pipeline
1. **Calculate Stock AR**: Using Fama-French 5-factor model
2. **Calculate Sector ETF AR**: Using same model
3. **Compute Deviation**: Stock AR - Sector AR
4. **Compare**: |Deviation| on extreme news days vs non-news days
5. **Test**: T-test for statistical significance

### Data
- **Period**: January 2019 - December 2024 (~1,500 trading days)
- **Stocks**: 6 major companies across different sectors
- **News Threshold**: |sentiment_polarity| > 0.95 (extreme sentiment only)
- **Sample**: ~940-1,070 extreme news days per stock

---

## Results by Sector

| Ticker | Sector | News Days | Deviation Increase | P-Value | Significant? |
|--------|--------|-----------|-------------------|---------|--------------|
| **NVDA** | Technology | 970 | **+11.3%** | 0.176 | ✗ No |
| **JPM** | Finance | 956 | **+8.4%** | 0.341 | ✗ No |
| **PFE** | Healthcare | 985 | **+0.1%** | 0.987 | ✗ No |
| **AMZN** | Consumer Discr. | 1,049 | **-7.5%** | 0.410 | ✗ No |
| **BA** | Industrials | 901 | **-9.0%** | 0.161 | ✗ No |
| **XOM** | Energy | 941 | **-22.2%** | **0.0001** | **✓ YES** |

### Aggregate Statistics
- **Average deviation increase**: -3.1%
- **Median deviation increase**: -3.7%
- **Statistically significant**: 1/6 (16.7%)

---

## Detailed Findings

### 1. Technology: NVDA
- **Model Fit**: Stock R² = 0.655, Sector R² = 0.924
- **Deviation**: 1.40% (news) vs 1.26% (non-news)
- **Increase**: +11.3% (**not significant**, p=0.176)
- **Interpretation**: NVDA shows slight increase in sector deviation on news days, but not statistically reliable

### 2. Finance: JPM
- **Model Fit**: Stock R² = 0.663, Sector R² = 0.890
- **Deviation**: 0.53% (news) vs 0.49% (non-news)
- **Increase**: +8.4% (**not significant**, p=0.341)
- **Interpretation**: JPM tracks sector closely; minimal news-specific deviation

### 3. Healthcare: PFE
- **Model Fit**: Stock R² = 0.214, Sector R² = 0.592
- **Deviation**: 0.97% (news) vs 0.96% (non-news)
- **Increase**: +0.1% (**not significant**, p=0.987)
- **Interpretation**: Essentially no difference; PFE's low R² suggests idiosyncratic factors beyond news

### 4. Energy: XOM ⚠️
- **Model Fit**: Stock R² = 0.529, Sector R² = 0.627
- **Deviation**: 0.45% (news) vs 0.58% (non-news)
- **Increase**: **-22.2%** (**SIGNIFICANT**, p=0.0001)
- **Interpretation**: **Paradoxical result** - XOM actually deviates LESS from its sector on extreme news days, suggesting oil sector moves as a bloc on major news

### 5. Consumer Discretionary: AMZN
- **Model Fit**: Stock R² = 0.624, Sector R² = 0.828
- **Deviation**: 0.90% (news) vs 0.97% (non-news)
- **Increase**: -7.5% (**not significant**, p=0.410)
- **Interpretation**: AMZN tracks sector closely regardless of news

### 6. Industrials: BA
- **Model Fit**: Stock R² = 0.420, Sector R² = 0.844
- **Deviation**: 1.38% (news) vs 1.51% (non-news)
- **Increase**: -9.0% (**not significant**, p=0.161)
- **Interpretation**: BA's volatility is not news-driven; likely operational/aerospace specific

---

## Key Insights

### 1. News Saturation
With **900-1,000+ extreme news days** per stock (70-85% of trading days), "extreme news" has become the **norm**, not the exception. The market may have already priced in this constant flow of high-sentiment information.

### 2. Sector Co-Movement
Particularly in **Energy (XOM)** and **Finance (JPM)**, stocks move with their sectors even on stock-specific news days. This suggests sector-wide factors dominate individual news impact.

### 3. Model Quality Matters
- **PFE** and **BA** have lower stock R² (<0.5), indicating the Fama-French model doesn't capture their risk well
- **Sector ETFs** consistently have higher R² (0.6-0.9), suggesting they're better explained by systematic factors
- Poor model fit may mask true news effects

### 4. Volatility ≠ News Impact
High-volatility stocks (NVDA, BA) don't necessarily show larger news-driven deviations. Much of their volatility appears to be:
- Sector-synchronized
- Driven by non-news factors (earnings, guidance, operations)

---

## Comparison to Previous Experiments

| Approach | AAPL/TSLA Original | This Multi-Sector Study |
|----------|-------------------|------------------------|
| **Stocks** | 2 (Tech) | 6 (Diverse sectors) |
| **News Events** | 150-200 per stock | 900-1,070 per stock |
| **Methodology** | Stock AR only | Stock AR - Sector AR (deviation) |
| **Significant Results** | Some evidence | 1/6 (16.7%) |
| **Conclusion** | Mixed | Weak/No evidence |

### Why Different Results?
1. **Sample Size**: More news = harder to find unique events
2. **Sector Control**: Removing sector effects reveals less stock-specific impact
3. **Broader Market**: Tech-heavy original sample vs diversified sectors
4. **Time Period**: 2019-2024 includes COVID and high-volatility periods

---

## Statistical Validity

### Strengths
✅ **Large sample size**: ~1,000+ observations per stock
✅ **Robust methodology**: Fama-French 5-factor model
✅ **Sector control**: Accounts for industry-wide movements
✅ **Consistent approach**: Same pipeline across all stocks

### Limitations
⚠️ **Model fit varies**: R² ranges from 0.21 (PFE) to 0.66 (JPM)
⚠️ **News saturation**: 70-85% of days have "extreme" news
⚠️ **Threshold selection**: |pol| > 0.95 may still capture too much
⚠️ **Correlation weak**: No consistent sentiment → deviation relationship

---

## Implications for News-Based Trading

### For Investors
1. **Extreme news is noisy**: High sentiment scores don't reliably predict stock-specific moves
2. **Sector matters more**: Stocks often move with their sector regardless of individual news
3. **Context required**: Raw sentiment without fundamental analysis is insufficient

### For Researchers
1. **Event saturation problem**: Need stricter filters or better event classification
2. **Model selection**: Consider stock-specific factor models beyond Fama-French
3. **News quality over quantity**: Focus on genuinely unexpected events (M&A, regulatory, disasters)

### For Practitioners
1. **Don't trade on sentiment alone**: Combine with:
   - Fundamental analysis
   - Technical indicators
   - Sector momentum
2. **Consider sector ETF trades**: If sector moves as a bloc, trade the sector instead
3. **Focus on genuine surprises**: Pre-announcement leaks, unexpected guidance, etc.

---

## Recommendations for Future Work

### 1. Stricter Event Selection
- Use **|polarity| > 0.99** or top 1% of sentiment scores
- Filter by **news surprise** (actual vs expected)
- Focus on **specific event types** (FDA approvals, earnings surprises, M&A)

### 2. Enhanced Methodology
- **Event study with longer windows**: [-1, +3] to capture delayed reactions
- **Intraday data**: Measure immediate reaction to news release
- **Volume analysis**: High deviation + high volume = more reliable signal

### 3. Alternative Approaches
- **Text classification**: Use NLP to categorize news by type/relevance
- **Causal inference**: Difference-in-differences or synthetic control
- **Machine learning**: Train models on features beyond just sentiment polarity

### 4. Additional Sectors
- **Technology** (NVDA) showed highest increase → expand tech sample
- **Commodities**: Gold miners, utilities (less correlated to broader market)
- **Small caps**: May be more news-sensitive than large caps

---

## Conclusion

This multi-sector analysis **does not support** the hypothesis that extreme sentiment news causes measurable stock-specific deviations from sector performance. The evidence suggests:

1. **News is priced efficiently**: Markets incorporate even "extreme" sentiment rapidly
2. **Sectors move together**: Systematic factors dominate idiosyncratic news
3. **Saturation problem**: Too many "extreme" news days dilute signal

### Final Verdict
**News sentiment alone is insufficient for predicting abnormal returns.** Successful news-based strategies likely require:
- More sophisticated event classification
- Integration with fundamental data
- Focus on genuinely unexpected information
- Sector and market timing considerations

---

## Files Generated

### Output Directory Structure
```
03-output/results/multi_sector_deviation/
├── sector_comparison.csv              # Aggregate results
├── multi_sector_comparison.png        # Visualization
├── AMZN/
│   ├── deviations.csv                # Daily deviation data
│   └── summary.csv                   # Stock summary
├── BA/
│   ├── deviations.csv
│   └── summary.csv
├── JPM/
│   ├── deviations.csv
│   └── summary.csv
├── NVDA/
│   ├── deviations.csv
│   └── summary.csv
├── PFE/
│   ├── deviations.csv
│   └── summary.csv
└── XOM/
    ├── deviations.csv
    └── summary.csv
```

### Key Files
- **sector_comparison.csv**: Cross-sector statistics and significance tests
- **multi_sector_comparison.png**: 6-panel visualization of results
- **{TICKER}/deviations.csv**: Daily deviation, news tags, sentiment for each stock
- **EXPERIMENT_SUMMARY.md**: This document

---

**Analysis Date**: October 12, 2025
**Python Scripts**: `11_multi_sector_deviation_analysis.py`
**Data Sources**: EODHD (news), Yahoo Finance (prices), Kenneth French (factors)
