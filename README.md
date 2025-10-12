# 📰 News Impact on Stock Returns: An Event Study Analysis

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Research Complete](https://img.shields.io/badge/status-research%20complete-success.svg)]()

> **A comprehensive empirical analysis examining whether financial news events create abnormal stock returns using rigorous event study methodology based on the Fama-French five-factor model.**

---

## 🎯 Project Overview

This research project investigates the **impact of financial news on stock returns** for Apple Inc. (AAPL) and Tesla Inc. (TSLA) using a traditional event study framework. By analyzing hundreds of thousands of news articles and thousands of trading days, we test whether public news creates exploitable trading opportunities.

### Key Research Question

**Do major news events create statistically significant abnormal returns that investors can exploit?**

**Answer: NO** - Our comprehensive analysis found no statistical evidence that news creates abnormal returns, strongly supporting market efficiency.

---

## 📊 Key Findings

### Summary Results

| Metric | AAPL | TSLA | Interpretation |
|--------|------|------|----------------|
| **Event Days** | 33 (3.2%) | 23 (1.6%) | Optimal density for event studies |
| **Mean AR Difference** | +0.11% | +0.02% | Economically negligible |
| **Effect Size (Cohen's d)** | 0.011 | 0.006 | Negligible effects |
| **Significant Tests** | 0/5 | 0/5 | No statistical evidence |
| **Model R²** | 0.774 | 0.434 | Excellent to moderate fit |

### Main Conclusions

1. ✅ **No Statistical Evidence** - Neither AAPL nor TSLA showed significant abnormal returns on news days
2. ✅ **Market Efficiency Supported** - Results strongly support the Efficient Market Hypothesis (EMH)
3. ✅ **No Trading Opportunities** - Public news does not create exploitable patterns
4. ✅ **Excellent Model Fit** - Fama-French factors explain 77% (AAPL) and 43% (TSLA) of returns

> **Bottom Line**: By the time news becomes public, it's already fully reflected in stock prices. The market is informationally efficient for large-cap technology stocks.

---

## 🏗️ Project Structure

```
news-stock-prediction/
│
├── 📂 01-data/                    # Data files (raw and processed)
│   ├── *_stock_data.csv          # Stock price data (AAPL, TSLA)
│   ├── *_news_raw.csv            # Raw news articles
│   ├── *_improved_events.csv     # Filtered event dates
│   ├── fama_french_factors.csv   # Fama-French 5 factors
│   └── README.md                 # Data documentation
│
├── 📂 02-scripts/                 # Analysis code
│   ├── 00_data_acquisition.py    # Data collection
│   ├── 01_data_loader.py         # Data loading utilities
│   ├── 02_beta_estimation.py     # Factor model estimation
│   ├── 03_abnormal_returns.py    # AR calculation
│   ├── 04_statistical_tests.py   # Hypothesis testing
│   ├── 05_main_analysis.py       # Main pipeline
│   ├── create_simple_presentation.py  # Visualization generator
│   └── README.md                 # Code documentation
│
├── 📂 03-output/                  # Results and visualizations
│   ├── AAPL_improved_study/      # AAPL analysis results
│   ├── TSLA_improved_study/      # TSLA analysis results
│   ├── presentation/             # Presentation materials
│   │   ├── overview_comparison.png
│   │   ├── AAPL_detailed_analysis.png
│   │   ├── TSLA_detailed_analysis.png
│   │   ├── news_characteristics.png
│   │   └── DETAILED_PRESENTATION_DOCUMENT.md
│   └── README.md                 # Results documentation
│
├── 📂 docs/                       # Additional documentation
│   ├── METHODOLOGY.md            # Detailed methodology
│   ├── DATA_DESCRIPTION.md       # Data documentation
│   └── RESULTS_INTERPRETATION.md # How to interpret results
│
├── 📄 README.md                   # This file
├── 📄 SETUP_AND_RUN.md           # Setup instructions
├── 📄 requirements.txt           # Python dependencies
├── 📄 .gitignore                 # Git ignore rules
└── 📄 .env.template              # API key template

```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (tested on 3.12)
- **Git** (for cloning)
- **API Keys** (optional, for data collection):
  - [EODHD Financial News API](https://eodhistoricaldata.com/)
  - [Alpha Vantage](https://www.alphavantage.co/) (alternative)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/news-stock-prediction.git
cd news-stock-prediction

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys (optional, for data collection)
cp .env.template .env
# Edit .env and add your API keys

# 5. Run the analysis
cd 02-scripts
python 05_main_analysis.py
```

### Using Pre-collected Data

If you don't want to collect data yourself, the repository includes pre-processed data:

```bash
cd 02-scripts
python 05_main_analysis.py  # Uses existing data in 01-data/
```

---

## 📖 Methodology

### Event Study Framework

This analysis employs a **traditional event study methodology**, the gold standard for measuring market impact of specific events.

#### 1. Factor Model Estimation

**Fama-French Five-Factor Model**:
```
R_i,t - R_f,t = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + ε_t
```

**Factors**:
- **Mkt-RF**: Market risk premium
- **SMB**: Small Minus Big (size factor)
- **HML**: High Minus Low (value factor)
- **RMW**: Robust Minus Weak (profitability factor)
- **CMA**: Conservative Minus Aggressive (investment factor)

#### 2. Abnormal Returns Calculation

**Abnormal Return (AR)** = Actual Return - Expected Return

Expected return is predicted from the factor model given the day's factor realizations.

#### 3. Statistical Testing

Five comprehensive tests:
1. **One-sample t-test** (event days): Tests if mean AR = 0
2. **One-sample t-test** (non-event days): Quality check
3. **Welch's t-test**: Compares event vs non-event days
4. **F-test**: Tests variance differences
5. **OLS Regression**: Measures news effect with dummy variable

**Significance Level**: α = 0.05 (95% confidence)

### News Filtering Pipeline

Raw news data filtered through 5 stages:

```
Stage 1: Sentiment Analysis
  └─> Filter: |Polarity| > 0.5 (strong sentiment)

Stage 2: Content Categorization
  └─> Filter: Priority categories (earnings, products, executive)

Stage 3: Volume-Based Filtering
  └─> Filter: Top 10% of days by article count

Stage 4: One Event Per Day
  └─> Select: Highest priority + strongest sentiment

Stage 5: Trading Day Alignment
  └─> Remove: Non-trading days (weekends, holidays)
```

**Result**:
- AAPL: 738,103 → 33 events (99.995% reduction)
- TSLA: 1,407,023 → 23 events (99.998% reduction)

---

## 📈 Results

### AAPL (Apple Inc.)

**Sample**: 1,025 trading days (2020-2024)
- Event Days: 33 (3.2%)
- Non-Event Days: 992 (96.8%)

**Abnormal Returns**:
```
Event Days:
  Mean AR:        +0.0869%
  Std Dev:         1.151%
  Range:          -1.40% to +4.04%

Non-Event Days:
  Mean AR:        -0.0200%
  Std Dev:         1.009%
  Range:          -4.46% to +8.20%

Difference:       +0.1067% (negligible)
```

**Statistical Tests**: 0/5 significant (all p > 0.24)

**Effect Size**: Cohen's d = 0.011 (negligible)

**Factor Model**: R² = 0.774 (excellent fit)

### TSLA (Tesla Inc.)

**Sample**: 1,422 trading days (2019-2024)
- Event Days: 23 (1.6%)
- Non-Event Days: 1,399 (98.4%)

**Abnormal Returns**:
```
Event Days:
  Mean AR:        -0.0227%
  Std Dev:         3.953%
  Range:          -8.24% to +12.84%

Non-Event Days:
  Mean AR:        -0.0426%
  Std Dev:         3.338%
  Range:         -21.48% to +21.11%

Difference:       +0.0199% (negligible)
```

**Statistical Tests**: 0/5 significant (all p > 0.20)

**Effect Size**: Cohen's d = 0.006 (negligible)

**Factor Model**: R² = 0.434 (moderate fit, TSLA more volatile)

---

## 🎨 Visualizations

High-resolution (300 DPI) publication-quality visualizations available in [`03-output/presentation/`](03-output/presentation/):

### 1. Overview Comparison
![Overview](03-output/presentation/overview_comparison.png)
*Side-by-side comparison of AAPL vs TSLA across 6 key metrics*

### 2. Detailed Analysis Dashboards
- **AAPL**: Distribution analysis, time series, factor loadings, statistical tests
- **TSLA**: Distribution analysis, time series, factor loadings, statistical tests

### 3. News Characteristics
![News](03-output/presentation/news_characteristics.png)
*Sentiment distribution, content length, data coverage*

---

## 💡 Key Insights

### Why No News Impact?

#### 1. Market Efficiency (Primary Explanation)
- **Algorithmic Trading**: Reacts in milliseconds, far faster than human traders
- **Pre-announcement Effects**: Informed traders position before official news
- **Institutional Advantage**: Bloomberg terminals get news seconds before public

#### 2. Event Identification Challenges
- **Timing Issues**: News articles published after market reaction
- **Publication Lag**: Article timestamp ≠ information availability
- **Multi-day Effects**: Reactions may span multiple days

#### 3. Data Granularity Limitations
- **Daily Data**: Misses intraday reactions (first minutes/hours critical)
- **Overnight Gaps**: After-hours news creates opening gaps we don't capture
- **Averaging Effect**: Daily returns smooth out intraday volatility

### Implications for Investors

**Don't**:
- ❌ Trade based on public news articles
- ❌ Expect to profit from "breaking news"
- ❌ React emotionally to headlines

**Do**:
- ✅ Focus on long-term factor exposures
- ✅ Maintain diversified portfolios
- ✅ Use news for context, not signals
- ✅ Minimize trading costs

**Trading Reality Check**:
- Expected return from news trading: ~0%
- Transaction costs: -0.1% to -0.5% per roundtrip
- **Net result**: Loses money

---

## 📚 Documentation

- **[SETUP_AND_RUN.md](SETUP_AND_RUN.md)**: Detailed setup and execution guide
- **[Presentation Document](03-output/presentation/DETAILED_PRESENTATION_DOCUMENT.md)**: 50+ page comprehensive analysis report
- **[01-data/README.md](01-data/README.md)**: Data description and sources
- **[02-scripts/README.md](02-scripts/README.md)**: Code documentation
- **[03-output/README.md](03-output/README.md)**: Results interpretation guide

---

## 🛠️ Technical Stack

### Core Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12+ |
| **Data Analysis** | pandas, numpy |
| **Statistical Modeling** | statsmodels, scipy |
| **Visualization** | matplotlib, seaborn |
| **Financial Data** | yfinance, pandas-datareader |
| **NLP/Sentiment** | vaderSentiment |

### Key Libraries

```python
pandas>=2.1.0           # Data manipulation
numpy>=1.26.0           # Numerical computing
statsmodels>=0.14.0     # Statistical models
scipy>=1.11.0           # Statistical tests
matplotlib>=3.8.0       # Plotting
seaborn>=0.13.0         # Statistical visualization
yfinance>=0.2.28        # Stock data
pandas-datareader>=0.10.0  # Fama-French factors
vaderSentiment>=3.3.2   # Sentiment analysis
```

---

## 🔬 Research Quality

### Strengths

✅ **Rigorous Methodology**: Traditional event study framework with established best practices
✅ **Comprehensive Testing**: 5 different statistical tests for robustness
✅ **Large Sample**: Analyzed 738K+ (AAPL) and 1.4M+ (TSLA) news articles
✅ **Clean Event Identification**: Multi-stage filtering ensures high-quality events
✅ **Excellent Model Fit**: Fama-French factors explain 77% (AAPL) of returns
✅ **Reproducible**: All code, data, and analysis fully documented

### Limitations

⚠️ **Daily Data**: Cannot capture intraday reactions (first minutes/hours)
⚠️ **Limited Sample**: Only 2 stocks (both large-cap tech)
⚠️ **Time Period**: 2019-2024 only (results may not generalize)
⚠️ **Event Identification**: Filtering may miss some material events
⚠️ **Sentiment Analysis**: VADER may not capture all financial nuance

### Future Research

- **Intraday Analysis**: Use tick-by-tick data to capture immediate reactions
- **Social Media**: Incorporate Twitter/Reddit for earlier signals
- **Machine Learning**: Better event identification and prediction
- **Cross-Sectional**: Analyze broader sample (S&P 500, Russell 3000)
- **Event Types**: Deep dive into earnings, M&A, product launches separately

---

## 📄 Citation

If you use this research or code, please cite:

```bibtex
@misc{news_stock_prediction_2024,
  title={News Impact on Stock Returns: An Event Study Analysis},
  author={[Your Name]},
  year={2024},
  institution={University of Southern California},
  course={DSO 585 - Data-Driven Consulting},
  url={https://github.com/yourusername/news-stock-prediction}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black 02-scripts/
isort 02-scripts/
```

---

## 👥 Authors & Acknowledgments

### Authors
- **[Your Name]** - *Principal Investigator* - [GitHub](https://github.com/yourusername)

### Acknowledgments
- **Professor [Name]** - Course instructor (DSO 585, USC Marshall)
- **Kenneth French** - Fama-French factor data
- **EODHD** - Financial news data API
- **Yahoo Finance** - Stock price data

### References

**Seminal Papers**:
1. Fama, E. F., et al. (1969). "The Adjustment of Stock Prices to New Information." *International Economic Review*
2. Ball, R., & Brown, P. (1968). "An Empirical Evaluation of Accounting Income Numbers." *Journal of Accounting Research*
3. Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model." *Journal of Financial Economics*

---

## 📞 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **Project Link**: [https://github.com/yourusername/news-stock-prediction](https://github.com/yourusername/news-stock-prediction)

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/news-stock-prediction&type=Date)](https://star-history.com/#yourusername/news-stock-prediction&Date)

---

## 📊 Project Status

- [x] **Phase 1**: News Impact Detection - **COMPLETE**
- [ ] **Phase 2**: Sentiment Classification - *Planned*
- [ ] **Phase 3**: Magnitude Prediction - *Planned*

**Last Updated**: October 11, 2024
**Status**: Research Complete, Paper in Progress

---

<div align="center">

**[⬆ Back to Top](#-news-impact-on-stock-returns-an-event-study-analysis)**

Made with ❤️ at USC Marshall School of Business

</div>
