# 📈 Output Directory

This directory contains all analysis results, visualizations, and presentation materials.

## 📁 Directory Structure

```
03-output/
├── 📊 Analysis Results (by Study)
│   ├── AAPL_improved_study/          # Final AAPL analysis
│   ├── TSLA_improved_study/          # Final TSLA analysis
│   ├── AAPL_major_events/            # AAPL major events only
│   ├── TSLA_major_events/            # TSLA major events only
│   └── filtered_analysis/            # Various filtering experiments
│
└── 🎨 Presentation Materials
    └── presentation/
        ├── overview_comparison.png
        ├── AAPL_detailed_analysis.png
        ├── TSLA_detailed_analysis.png
        ├── news_characteristics.png
        └── DETAILED_PRESENTATION_DOCUMENT.md
```

## 📊 Analysis Result Files

Each analysis directory (e.g., `AAPL_improved_study/`) contains:

### Core Results

**`analysis_summary.csv`**
- One-row summary of the entire analysis
- Columns: `total_days, news_days, non_news_days, mean_ar_news, mean_ar_non_news, std_ar_news, std_ar_non_news, avg_r_squared, significant_tests, total_tests`

**`abnormal_returns.csv`**
- Daily abnormal returns for all trading days
- Columns: `Date, Abnormal_Return, News_Day, Expected_Return, Actual_Return`
- News_Day: 1 = event day, 0 = non-event day

**`cumulative_abnormal_returns.csv`**
- Cumulative abnormal returns over time
- Columns: `News_Date, Window_Start, Window_End, CAR, N_days, Mean_AR, Std_AR`

**`ar_statistics.csv`**
- Summary statistics by group
- Groups: News Days, Non-News Days
- Statistics: N, Mean, Std, Min, Max, Median, Percentiles

**`statistical_tests.csv`**
- Results from all 5 hypothesis tests
- Columns: `Test, N, Mean, Std, t_statistic, p_value, Significant, ...`
- Tests: One-sample t (2x), Welch's t, F-test, OLS Regression

**`beta_estimates.csv`**
- Factor model beta coefficients
- Columns: `Date, Alpha, R_squared, N_obs, Residual_Std, Beta_Mkt-RF, Beta_SMB, Beta_HML, Beta_RMW, Beta_CMA`

**`beta_stability.csv`**
- Rolling window beta estimates (if computed)
- Shows how betas change over time

**`subperiod_analysis.csv`**
- Analysis split by time periods
- Tests robustness across different market conditions

### Visualizations

**`analysis_summary.png`**
- Multi-panel summary visualization
- Shows key metrics and distributions

## 📊 Key Results Summary

### AAPL_improved_study

**Sample Characteristics**:
```
Total Days:              1,025
Event Days:              33 (3.2%)
Non-Event Days:          992 (96.8%)
Date Range:              2020-08-24 to 2024-10-10
```

**Abnormal Returns**:
```
Event Days:
  Mean AR:               +0.0869%
  Std Dev:               1.151%
  Range:                 -1.40% to +4.04%

Non-Event Days:
  Mean AR:               -0.0200%
  Std Dev:               1.009%
  Range:                 -4.46% to +8.20%

Difference:              +0.1067% (negligible)
```

**Statistical Tests**: 0/5 significant (p-values: 0.25-0.67)

**Factor Model**: R² = 0.774 (excellent fit)

**Effect Size**: Cohen's d = 0.011 (negligible)

---

### TSLA_improved_study

**Sample Characteristics**:
```
Total Days:              1,422
Event Days:              23 (1.6%)
Non-Event Days:          1,399 (98.4%)
Date Range:              2019-01-02 to 2024-10-10
```

**Abnormal Returns**:
```
Event Days:
  Mean AR:               -0.0227%
  Std Dev:               3.953%
  Range:                 -8.24% to +12.84%

Non-Event Days:
  Mean AR:               -0.0426%
  Std Dev:               3.338%
  Range:                 -21.48% to +21.11%

Difference:              +0.0199% (negligible)
```

**Statistical Tests**: 0/5 significant (p-values: 0.20-0.98)

**Factor Model**: R² = 0.434 (moderate fit)

**Effect Size**: Cohen's d = 0.006 (negligible)

---

## 🎨 Presentation Materials

Located in `presentation/` subdirectory.

### Visualizations (300 DPI PNG)

**1. overview_comparison.png**
- Side-by-side comparison of AAPL vs TSLA
- 6 panels: Sample composition, Mean AR, Difference, Volatility, R², Significance
- Dimensions: 5000×3000 pixels
- Format: PNG with transparency

**2. AAPL_detailed_analysis.png**
- Comprehensive AAPL analysis dashboard
- 6 panels: AR distribution, time series, box plot, factor loadings, statistics table, tests summary
- Dimensions: 5400×4200 pixels

**3. TSLA_detailed_analysis.png**
- Comprehensive TSLA analysis dashboard
- Same structure as AAPL
- Dimensions: 5400×4200 pixels

**4. news_characteristics.png**
- News data analysis
- 4 panels: Sentiment distribution, content length, sentiment components, coverage summary
- Dimensions: 5400×3600 pixels

### Documentation

**DETAILED_PRESENTATION_DOCUMENT.md**
- Comprehensive 50+ page analysis report
- Sections:
  1. Executive Summary
  2. Methodology
  3. Data Description
  4. News Filtering Process
  5. AAPL Detailed Results
  6. TSLA Detailed Results
  7. Comparative Analysis
  8. Statistical Significance
  9. Discussion & Interpretation
  10. Conclusions
  11. Limitations & Future Research
  12. Technical Appendix

- Word count: ~25,000 words
- Format: Markdown with code blocks, tables, equations

---

## 📖 How to Interpret Results

### Understanding Abnormal Returns

**Abnormal Return (AR)**: The return NOT explained by systematic risk factors
```
AR = Actual Return - Expected Return
AR = (R_actual - R_f) - (β₁·Mkt-RF + β₂·SMB + β₃·HML + β₄·RMW + β₅·CMA)
```

**Interpretation**:
- **AR > 0**: Stock outperformed expectations (positive surprise)
- **AR < 0**: Stock underperformed expectations (negative surprise)
- **AR ≈ 0**: Stock performed as expected (no surprise)

**On News Days**: We expect AR ≠ 0 if news has impact
**On Non-News Days**: We expect AR ≈ 0 (confirms model quality)

### Statistical Test Results

**Test 1: One-Sample t-test (Event Days)**
- **Question**: Do event days have non-zero abnormal returns?
- **H₀**: Mean AR on event days = 0
- **Interpretation**:
  - Significant (p < 0.05) → News creates abnormal returns
  - Not significant (p ≥ 0.05) → No evidence of news impact

**Test 2: One-Sample t-test (Non-Event Days)**
- **Question**: Do non-event days have zero abnormal returns (quality check)?
- **H₀**: Mean AR on non-event days = 0
- **Interpretation**:
  - Not significant (p ≥ 0.05) → ✓ Model works well (expected)
  - Significant (p < 0.05) → ✗ Model has problems

**Test 3: Welch's t-test**
- **Question**: Are event days different from non-event days?
- **H₀**: Mean AR(event) = Mean AR(non-event)
- **Interpretation**:
  - Significant (p < 0.05) → News days differ from normal days
  - Not significant (p ≥ 0.05) → No detectable difference

**Test 4: F-test (Variance)**
- **Question**: Does news increase volatility?
- **H₀**: Var(event) = Var(non-event)
- **Interpretation**:
  - Significant (p < 0.05) → News increases volatility
  - Not significant (p ≥ 0.05) → Similar volatility

**Test 5: OLS Regression**
- **Model**: AR = β₀ + β₁·News_Dummy + ε
- **Question**: Does news indicator predict abnormal returns?
- **H₀**: β₁ = 0
- **Interpretation**:
  - Significant (p < 0.05) → News has measurable effect
  - Not significant (p ≥ 0.05) → News doesn't predict returns
  - R² → How much variance explained by news (0-1)

### Effect Size (Cohen's d)

Measures **practical significance** (not just statistical significance)

```
Cohen's d = (Mean₁ - Mean₂) / Pooled_SD
```

**Interpretation**:
- |d| < 0.2 → **Negligible** effect
- 0.2 ≤ |d| < 0.5 → Small effect
- 0.5 ≤ |d| < 0.8 → Medium effect
- |d| ≥ 0.8 → Large effect

**Our Results**:
- AAPL: d = 0.011 → Negligible (95% below "small")
- TSLA: d = 0.006 → Negligible (97% below "small")

**Conclusion**: Even if tests were significant, effect is too small to matter.

### Factor Model R²

Measures how well Fama-French factors explain returns

```
R² = 1 - (SS_residual / SS_total)
```

**Interpretation**:
- R² = 0.00 → Factors explain 0% of variation (bad)
- R² = 0.50 → Factors explain 50% of variation (okay)
- R² = 0.70 → Factors explain 70% of variation (good)
- R² = 0.90 → Factors explain 90% of variation (excellent)

**Our Results**:
- AAPL: R² = 0.774 → **Excellent** (77.4% explained)
- TSLA: R² = 0.434 → **Moderate** (43.4% explained)

**Why TSLA lower?**: More volatile, idiosyncratic, Elon Musk factor

---

## 🔍 Results Interpretation Guide

### Scenario: Significant News Impact (Hypothetical)

If we HAD found significant results, they would look like:

```
Event Days:
  Mean AR:               +2.50%    ← Large positive
  Std Dev:               3.20%

Non-Event Days:
  Mean AR:               +0.05%    ← Near zero
  Std Dev:               1.80%

Statistical Tests:
  ✓ One-sample t (event):    t=8.5, p<0.001  ← SIGNIFICANT
  ✓ Welch's t-test:          t=7.2, p<0.001  ← SIGNIFICANT
  ✓ F-test:                  F=3.2, p=0.002  ← SIGNIFICANT
  ✓ OLS Regression:          β=2.45%, p<0.001, R²=0.15  ← SIGNIFICANT

Effect Size: Cohen's d = 0.92 (large effect)

Interpretation: Strong evidence that news creates 2.5% abnormal returns
```

### Actual Results: No Significant Impact

What we actually found:

```
Event Days:
  Mean AR:               +0.09%    ← Tiny positive
  Std Dev:               1.15%

Non-Event Days:
  Mean AR:               -0.02%    ← Tiny negative
  Std Dev:               1.01%

Statistical Tests:
  ✗ One-sample t (event):    t=0.43, p=0.668  ← NOT significant
  ✗ Welch's t-test:          t=0.53, p=0.603  ← NOT significant
  ✗ F-test:                  F=1.30, p=0.249  ← NOT significant
  ✗ OLS Regression:          β=0.11%, p=0.553, R²=0.0004  ← NOT significant

Effect Size: Cohen's d = 0.011 (negligible)

Interpretation: No evidence that news creates abnormal returns
```

---

## 📊 Using Results for Further Analysis

### Export to Other Formats

**Convert to Excel**:
```python
import pandas as pd

# Read CSV
df = pd.read_csv('03-output/AAPL_improved_study/abnormal_returns.csv')

# Save as Excel
df.to_excel('AAPL_abnormal_returns.xlsx', index=False)
```

**Create Custom Visualizations**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
ar_df = pd.read_csv('03-output/AAPL_improved_study/abnormal_returns.csv')

# Custom plot
event_ar = ar_df[ar_df['News_Day']==1]['Abnormal_Return']
plt.hist(event_ar, bins=20)
plt.title('AAPL Event Day Abnormal Returns')
plt.xlabel('Abnormal Return')
plt.ylabel('Frequency')
plt.savefig('custom_plot.png', dpi=300)
```

### Statistical Software

**Import to R**:
```r
# Read CSV
ar_data <- read.csv("03-output/AAPL_improved_study/abnormal_returns.csv")

# Run custom analysis
library(dplyr)
summary_stats <- ar_data %>%
  group_by(News_Day) %>%
  summarise(
    mean_ar = mean(Abnormal_Return),
    sd_ar = sd(Abnormal_Return),
    n = n()
  )
```

**Import to Stata**:
```stata
* Import CSV
import delimited "03-output/AAPL_improved_study/abnormal_returns.csv", clear

* Run t-test
ttest abnormal_return if news_day==1, by(news_day)
```

---

## 🎓 Academic Use

### For Papers/Presentations

**Figure References**:
```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{03-output/presentation/overview_comparison.png}
  \caption{Comparative Analysis of AAPL vs TSLA News Impact}
  \label{fig:overview}
\end{figure}
```

**Table References**:
```latex
\begin{table}[h]
  \centering
  \caption{Statistical Test Results}
  \label{tab:tests}
  \begin{tabular}{lcc}
    \hline
    Test & AAPL & TSLA \\
    \hline
    One-sample t (event) & p=0.668 & p=0.978 \\
    Welch's t-test & p=0.603 & p=0.981 \\
    \hline
  \end{tabular}
\end{table}
```

### Citation

When using these results:
```bibtex
@misc{news_stock_prediction_2024,
  title={News Impact on Stock Returns: An Event Study Analysis},
  author={[Your Name]},
  year={2024},
  note={AAPL: 0/5 tests significant; TSLA: 0/5 tests significant},
  url={https://github.com/yourusername/news-stock-prediction}
}
```

---

## ⚠️ Important Notes

### Data Files (.gitignore)

Large result files are **NOT** committed to Git:
- `*.csv` files > 1MB
- `*.png` files (already in repo, but can be regenerated)

To regenerate all results:
```bash
cd 02-scripts
python 05_main_analysis.py
python create_simple_presentation.py
```

### Result Reproducibility

Results should be **exactly reproducible** if:
1. Using same input data
2. Same random seed (if applicable)
3. Same package versions

Small variations (<0.0001) may occur due to:
- Floating-point precision
- Different NumPy/SciPy versions
- OS differences

---

## 📞 Questions?

If results are unclear:
1. Read `DETAILED_PRESENTATION_DOCUMENT.md` for full explanation
2. Check `02-scripts/README.md` for methodology
3. Review main `README.md` for project overview

---

**Last Updated**: October 11, 2024
**Results Generated**: October 11, 2024
**Analysis Version**: 1.0 (Final)
