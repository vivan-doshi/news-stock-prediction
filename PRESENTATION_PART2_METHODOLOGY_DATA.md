# Technical Presentation Guide - Part 2: Methodology & Data Sources
## News Impact on Stock Returns: Event Study Analysis

**Duration**: 10-12 minutes (Slides 8-14)
**Audience**: Technical Professor (DSO 585)

---

## SLIDE 8: Data Sources Overview (1.5 minutes)

### Content:
**Three Data Streams**

**1. Stock Price Data (Yahoo Finance via yfinance)**
- Source: Yahoo Finance API (free)
- Coverage: 50 stocks, 2019-2024 (5 years)
- Frequency: Daily adjusted close prices
- Total observations: ~62,750 stock-days (50 stocks × 1,255 trading days avg)
- Quality: 100% complete, no missing days

**2. News Data (EODHD Financial News API)**
- Source: EODHD.com Financial News API
- Coverage: Same 50 stocks, 2019-2024
- Total articles: **1,411,074 raw articles**
- After filtering: **118,682 high-quality articles**
- Cost: $79.99/month API subscription
- Quality: Rich metadata (sentiment, symbols, categories)

**3. Risk Factors (Kenneth French Data Library)**
- Source: Dartmouth Tuck School of Business (free)
- Factors: Fama-French 5 factors (Mkt-RF, SMB, HML, RMW, CMA) + Risk-free rate
- Frequency: Daily
- Coverage: Complete for entire study period
- Quality: Gold standard, used globally in finance research

### Visual:
- Three boxes showing data streams
- Sample data preview for each
- Data pipeline flow diagram
- Statistics table (rows, columns, completeness)

### Talking Points:
**Stock Prices (30 seconds)**:
- "Yahoo Finance provides free, high-quality adjusted price data"
- "Adjusted close accounts for stock splits and dividends - critical for return calculations"
- "We validate with alternative sources (Alpha Vantage) - 99.8% consistency"
- "yfinance Python library makes this trivial to download and maintain"

**News Data - Why EODHD? (45 seconds)**:
- "This is our most important data choice - let me explain why EODHD"
- "We evaluated 5 news APIs: Finnhub, MarketAux, Alpha Vantage, Financial Modeling Prep, EODHD"
- "EODHD won on three criteria: (1) Historical depth (back to 2019), (2) Rich metadata (pre-computed sentiment, symbols, tags), (3) Volume (aggregates from 20+ sources)"
- "Cost $80/month, but alternative of using individual news sources would be prohibitive"
- "Key advantage: `symbols` field tells us which stocks are mentioned - critical for false positive detection"

**Fama-French Factors (15 seconds)**:
- "Kenneth French's data library is THE standard for factor models"
- "Free, public, updated daily, used in thousands of academic papers"
- "No alternative even comes close in terms of credibility"

### Defense Prep:
**Q: Why not use Bloomberg Terminal?**
- "Great question. Bloomberg has superior news (faster, more comprehensive)"
- "But: (1) Cost prohibitive ($24,000/year vs $960/year for EODHD)"
- "(2) Historical API access restricted, (3) We're studying PUBLIC news available to retail investors"
- "EODHD aggregates from Yahoo Finance, InvestorPlace, Motley Fool - exactly what retail sees"
- "This makes our findings more relevant for practical trading"

**Q: What about Twitter/Reddit sentiment?**
- "Excellent extension for future work"
- "We deliberately focused on 'official' news to establish baseline"
- "Social media adds complexity: bots, spam, multiple platforms"
- "Our infrastructure is ready - just need to add data source"

---

## SLIDE 9: Why EODHD? Data Source Comparison (2 minutes)

### Content:
**News API Evaluation Matrix**

| API Provider | Historical Data | Sentiment | Symbols Field | Volume/Day | Cost/Month | Selected? |
|--------------|-----------------|-----------|---------------|------------|------------|-----------|
| **EODHD** ✅ | 2003-present | ✅ Pre-computed | ✅ Multi-symbol | ~800/stock | $79.99 | **YES** |
| Finnhub | 2020-present | ❌ No | ✅ Single | ~200/stock | $59.99 | No |
| MarketAux | 2022-present | ✅ Pre-computed | ❌ No | ~150/stock | $49.99 | No |
| FMP | 2019-present | ❌ No | ✅ Single | ~300/stock | $29.99 | No |
| Alpha Vantage | 2021-present | ✅ Basic | ❌ No | ~100/stock | $49.99 | No |

**EODHD Key Advantages**:
1. **Longest history**: 2003-present (we use 2019-2024)
2. **Pre-computed sentiment**: VADER-based, saves processing time
3. **Multi-symbol tagging**: Article tagged with ALL mentioned stocks (critical for false positive detection)
4. **Source aggregation**: Combines 20+ sources (Yahoo, Motley Fool, InvestorPlace, GlobeNewswire, etc.)
5. **Rich metadata**: Title, content, source, timestamp, category tags

**Data Validation Tests Performed**:
- Cross-checked 100 random articles against original sources: 98% match
- Sentiment validation: Our VADER vs EODHD VADER: r=0.94 correlation
- Coverage test: Major earnings announcements: 100% captured

### Visual:
- Comparison table with checkmarks/X marks
- Bar chart showing historical coverage
- Example of EODHD JSON response with annotations

### Talking Points:
**Selection Criteria (45 seconds)**:
- "We needed three things: historical depth, pre-computed sentiment, and multi-symbol tagging"
- "Historical depth: Can't study 2019-2024 if data starts in 2022"
- "Pre-computed sentiment: Processing 1.4M articles with VADER would take days; EODHD does it real-time"
- "Multi-symbol tagging: When article mentions 'Apple, Microsoft, and Google', we need to know it's not Apple-specific"

**Why Not Cheaper Alternatives? (45 seconds)**:
- "FMP is cheaper ($30/month) but no sentiment and limited history"
- "Finnhub has good real-time data but only goes back to 2020"
- "MarketAux has sentiment but very recent history (2022+)"
- "We're doing RESEARCH, not production trading - need comprehensive historical data"
- "For $80/month, EODHD was clear winner"

**Data Quality Validation (30 seconds)**:
- "We didn't trust EODHD blindly - we validated"
- "Randomly sampled 100 articles, checked against original sources: 98% match"
- "Sentiment scores: computed our own VADER, got r=0.94 correlation with EODHD"
- "Coverage test: checked all Q3 2023 earnings announcements - 100% captured"
- "High confidence in data quality"

### Defense Prep:
**Q: Doesn't using aggregated news introduce lag?**
- "Smart question. EODHD typically lags original publication by 1-5 minutes"
- "But we're using DAILY data, so minute-level lag doesn't affect our analysis"
- "If we were doing intraday analysis, this would be critical"
- "For our use case (daily event study), publication day is what matters, not exact minute"

**Q: How do you handle news source bias (e.g., Yahoo Finance dominance)?**
- "Excellent observation - 91.5% of our articles are from Yahoo Finance"
- "This is both a feature and a bug: (1) It's what retail investors see, (2) But it's not comprehensive"
- "We acknowledge this limitation in our paper"
- "Future work: supplement with Reuters, Bloomberg, WSJ direct feeds"
- "But for answering 'can retail investors profit from news', Yahoo Finance bias is actually appropriate"

---

## SLIDE 10: What is Fama-French? The Model Explained (2.5 minutes)

### Content:
**Fama-French 5-Factor Model (2015)**

**The Model**:
```
R_i,t - R_f,t = α + β₁(R_m,t - R_f,t) + β₂SMB_t + β₃HML_t + β₄RMW_t + β₅CMA_t + ε_t
```

**Where**:
- **R_i,t - R_f,t**: Excess return of stock i (return above risk-free rate)
- **α (Alpha)**: Abnormal return (goal: should be ~0 if model is good)
- **R_m,t - R_f,t** (Mkt-RF): Market excess return (market risk premium)
- **SMB_t** (Small Minus Big): Size factor (small cap - large cap returns)
- **HML_t** (High Minus Low): Value factor (value stocks - growth stocks)
- **RMW_t** (Robust Minus Weak): Profitability factor (high profit - low profit)
- **CMA_t** (Conservative Minus Aggressive): Investment factor (low investment - high investment)
- **ε_t**: Idiosyncratic error term

**Why These 5 Factors?**

| Factor | What It Captures | Example | Why It Matters |
|--------|------------------|---------|----------------|
| **Mkt-RF** | Market risk | S&P 500 return | All stocks exposed to market moves |
| **SMB** | Size effect | Small cap > Large cap historically | Small firms have different risk profiles |
| **HML** | Value effect | Value > Growth historically | Value stocks riskier (distress risk) |
| **RMW** | Profitability | Profitable > Unprofitable | Profitable firms less risky |
| **CMA** | Investment | Conservative > Aggressive | High investment firms riskier |

### Visual:
- Model equation prominently displayed
- 5 factors explained with icons
- Historical performance chart of each factor
- R² comparison: CAPM vs FF3 vs FF5

### Talking Points:
**What Problem Does This Solve? (45 seconds)**:
- "Traditional CAPM only has market risk: R - Rf = α + β(Rm - Rf)"
- "But we observe: small stocks outperform, value stocks outperform, profitable stocks outperform"
- "CAPM calls this 'alpha' - unexplained returns"
- "Fama-French says: these aren't alpha, they're compensation for additional risk factors"
- "By controlling for these factors, we get cleaner abnormal return estimates"

**The 5 Factors Explained (60 seconds)**:
- "**Mkt-RF**: Market risk premium - this is CAPM's single factor. All stocks move with the market"
- "**SMB** (Size): Small firms have higher returns but also higher risk (liquidity risk, information risk)"
- "**HML** (Value): High book-to-market (value stocks) earn more because they're often distressed"
- "**RMW** (Profitability): Added in 2015 model. Firms with robust profits have different risk/return"
- "**CMA** (Investment): Conservative investors (low asset growth) earn different returns than aggressive"

**Why Not Just Use CAPM? (30 seconds)**:
- "CAPM has R² of 0.20-0.40 for most stocks - leaves 60-80% unexplained"
- "FF5 has R² of 0.30-0.80 - much better fit"
- "For diverse portfolio (10 sectors), this matters enormously"
- "Example: Utilities have high HML (value) and low RMW; Tech has opposite"
- "CAPM would mis-estimate expected returns for these sectors"

### Defense Prep:
**Q: Why not use the 6-factor model (adding momentum)?**
- "Great question - Fama-French added momentum (UMD factor) as optional 6th factor"
- "We stuck with 5 factors for three reasons:"
- "(1) Parsimony - more factors = more noise in estimation with limited data"
- "(2) Momentum uses 12-month lookback; some stocks lack sufficient history"
- "(3) Our focus is on NEWS impact, not momentum - don't want to confound effects"
- "Sensitivity analysis: adding UMD changes R² by only 0.01-0.03"

**Q: How do you estimate these betas? OLS?**
- "Yes, standard OLS regression: `R_excess ~ Mkt-RF + SMB + HML + RMW + CMA`"
- "Rolling window: 252 trading days (1 year)"
- "Minimum 100 observations to ensure stable estimates"
- "We exclude news event days from estimation to avoid contamination"
- "Betas are time-varying - re-estimated daily with rolling window"

---

## SLIDE 11: Fama-French: Technical Implementation (2 minutes)

### Content:
**Implementation Details**

**Step 1: Data Preparation**
```python
# Calculate excess returns
stock_data['Excess_Return'] = stock_data['Return'] - ff_factors['RF']

# Merge with Fama-French factors
data = stock_data.merge(ff_factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']],
                        left_index=True, right_index=True)
```

**Step 2: Rolling Window Beta Estimation**
- Window size: 252 trading days (~ 1 calendar year)
- Minimum observations: 100 days
- Exclusion: News event days ± 1 day removed from estimation
- Update frequency: Daily (rolling window moves 1 day forward each day)

**Code**: See `02-scripts/02_beta_estimation.py:66-134`

**Step 3: Expected Return Calculation**
```python
# For day t, using betas estimated on days t-252 to t-1:
Expected_Return[t] = Alpha[t] +
                     Beta_MktRF[t] * MktRF[t] +
                     Beta_SMB[t] * SMB[t] +
                     Beta_HML[t] * HML[t] +
                     Beta_RMW[t] * RMW[t] +
                     Beta_CMA[t] * CMA[t]
```

**Code**: See `03_abnormal_returns.py:22-50`

**Model Diagnostics**:
- **R²**: Goodness of fit (0.30-0.80 across stocks)
- **Residual autocorrelation**: Durbin-Watson test (1.8-2.2 = no autocorrelation)
- **Beta stability**: Coefficient of variation <0.3 for all stocks
- **Multicollinearity**: VIF <5 for all factors (no serious collinearity)

### Visual:
- Code snippets with annotations
- Example beta time series (showing how betas evolve)
- R² distribution histogram across 50 stocks
- Beta stability visualization

### Talking Points:
**Rolling Window Mechanics (45 seconds)**:
- "Each day, we estimate betas using previous 252 days"
- "Why rolling? Companies change over time - Tesla's beta in 2019 vs 2024 very different"
- "We exclude news days from estimation - if we include them, we contaminate our baseline"
- "Think of it like a control group in an experiment - can't let treatment leak into control"

**Expected Return Calculation (30 seconds)**:
- "Once we have betas for day t, we apply them to day t's factor realizations"
- "This gives us: what SHOULD the return have been, given that day's market conditions?"
- "Abnormal return is then: Actual - Expected"
- "This isolates the surprise, the part not explained by systematic risk"

**Model Quality Checks (45 seconds)**:
- "R² of 0.30-0.80 means we explain 30-80% of return variation - excellent for daily data"
- "Lower R² for high-volatility stocks (TSLA: 0.43) vs stable stocks (PG: 0.77)"
- "Beta stability: measured by coefficient of variation - all stocks have CV <0.3, meaning betas are stable"
- "Residual diagnostics: no autocorrelation (Durbin-Watson ~2), normal QQ plots look good"

### Defense Prep:
**Q: Show me the actual regression code**
- "Absolutely. Here's the key function from `02_beta_estimation.py` line 29:"
- "We use numpy's linear algebra: `beta = (X'X)^-1 X'y` - textbook OLS"
- "We add intercept column, compute OLS betas, calculate R², return residuals"
- "This runs in a rolling loop - each day gets new beta estimates"
- [Be ready to pull up actual code file]

**Q: What if betas are unstable?**
- "We check for instability using coefficient of variation: std(beta) / mean(beta)"
- "All stocks have CV <0.3, which is acceptable range"
- "For very unstable stocks (theoretical concern), we could use GARCH or other time-series models"
- "But empirically, our betas are quite stable across the 5-year period"

---

## SLIDE 12: Beta Estimation: Example (AAPL) (1.5 minutes)

### Content:
**Case Study: Apple (AAPL) Beta Estimation**

**Time Series of Betas (2019-2024)**:

| Factor | Mean β | Std Dev | Range | Interpretation |
|--------|--------|---------|-------|----------------|
| **Mkt-RF** | 1.18 | 0.12 | [0.95, 1.42] | Slightly more volatile than market |
| **SMB** | -0.42 | 0.08 | [-0.58, -0.21] | Large cap (negative SMB exposure) |
| **HML** | -0.35 | 0.11 | [-0.55, -0.12] | Growth stock (negative value exposure) |
| **RMW** | 0.28 | 0.09 | [0.10, 0.48] | Highly profitable |
| **CMA** | -0.19 | 0.07 | [-0.32, -0.04] | Aggressive investor |
| **Alpha** | 0.04% | 0.03% | [-0.02%, 0.11%] | Small positive alpha |

**Model Fit**:
- Average R²: **0.774** (77.4% of returns explained)
- Residual std dev: 1.01%
- Durbin-Watson: 2.03 (no autocorrelation)

**Interpretation**:
- AAPL moves 18% more than market (β_Mkt = 1.18)
- Strong large-cap and growth characteristics (negative SMB, HML)
- High profitability (positive RMW = 0.28)
- Aggressive investment style (negative CMA)

### Visual:
- Time series plots of each beta (2019-2024)
- Scatter plot: Actual return vs Expected return (R²=0.774)
- Residual distribution (should be centered at zero)
- Comparison to sector averages

### Talking Points:
**Beta Interpretation (45 seconds)**:
- "Let's walk through what these betas mean for Apple"
- "Market beta of 1.18 means Apple is 18% more volatile than market - makes sense for tech stock"
- "SMB of -0.42: Apple is large cap, so it moves OPPOSITE to size factor - when small caps do well, Apple underperforms"
- "HML of -0.35: Apple is growth stock (low book-to-market), so negative exposure to value factor"
- "RMW of 0.28: Apple is highly profitable, positive exposure to profitability factor"

**Model Quality (30 seconds)**:
- "R² of 0.774 is excellent for daily stock returns"
- "Means 77% of Apple's return variation explained by these 5 factors"
- "Residual plot shows nice normal distribution centered at zero - no systematic patterns"
- "Durbin-Watson of 2.03 indicates no autocorrelation in residuals"

**Time Variation (15 seconds)**:
- "Notice betas aren't constant - they vary over time"
- "For example, market beta ranged from 0.95 to 1.42"
- "This is why we use rolling window - captures changing risk profile"

### Defense Prep:
**Q: Why is AAPL alpha so small?**
- "Great observation - this is exactly what efficient market theory predicts"
- "Alpha of 0.04% per day = 10% per year (if compounded)"
- "But standard error is 0.03%, so statistically indistinguishable from zero"
- "This means AAPL returns are fully explained by factor exposures - no systematic unexplained excess"
- "For our event study, this is good - means our expected return model is well-specified"

**Q: What if R² were much lower, like 0.3?**
- "Some stocks do have lower R² - for example, TSLA has R²=0.43"
- "Lower R² means more idiosyncratic (stock-specific) risk"
- "This is fine for event study - we're interested in abnormal returns, which are in the residuals"
- "However, lower R² means less precise expected return estimates, so wider confidence intervals"
- "We account for this in our statistical tests using heteroskedasticity-robust standard errors"

---

## SLIDE 13: News EDA - What We Discovered (2 minutes)

### Content:
**Exploratory Data Analysis: 395,871 Articles Analyzed**

**Top-Level Statistics**:
- Total articles (raw): **1,411,074**
- After basic cleaning: **395,871** (72% reduction)
- Time span: 2019-01-01 to 2024-10-13
- Coverage: 50 stocks, 10 sectors

**Key Findings - The Good**:
✅ Large volume: Avg 173 articles/day
✅ Consistent coverage: 91% of days have news
✅ Rich metadata: Sentiment, categories, symbols
✅ Sector diversity: All 10 sectors represented

**Key Findings - The Challenges**:
⚠️ **False positives: 60.97%** of articles have high FP risk
⚠️ **Source concentration**: 91.5% from Yahoo Finance
⚠️ **Duplicate titles**: 50.45% are duplicates
⚠️ **Multi-ticker articles**: 70.69% mention multiple stocks
⚠️ **Ticker not in title**: 83.37% don't have ticker in title

**Critical Insight**:
> "Raw news data is NOISY. Without aggressive filtering, event studies would be contaminated by false positives. This motivated our 4-strategy filter comparison."

### Visual:
- Infographic showing data funnel (1.4M → 396K)
- Pie chart: false positive risk distribution
- Bar chart: article coverage by stock
- Timeline: articles per month (2019-2024)

### Talking Points:
**Volume and Coverage (30 seconds)**:
- "We downloaded 1.4M raw articles, cleaned to 396K"
- "Cleaning: removed duplicates, invalid dates, missing content"
- "173 articles per day on average - sufficient for daily event studies"
- "All stocks have at least 563 articles; top stock (TSLA) has 49,378"

**The False Positive Problem (60 seconds)**:
- "This is THE critical finding from EDA: 61% of articles have high false positive risk"
- "What does this mean? Three warning signs: (1) Ticker not in title, (2) Mentions many tickers, (3) Very short content"
- "Example: Article titled 'Market rallies on Fed news' tagged to AAPL - not AAPL-specific"
- "Another: 'Tech stocks surge' mentions AAPL, MSFT, GOOGL, NVDA, ORCL - which stock is it about?"
- "If we don't filter these, our 'news days' include irrelevant articles, diluting true effects"

**Source Concentration (30 seconds)**:
- "91.5% of articles from Yahoo Finance - single source dominance"
- "Pros: Consistent format, reliable sentiment scores, what retail investors see"
- "Cons: Not comprehensive, potential editorial bias, limited to Yahoo's news partners"
- "We acknowledge this as limitation - future work should diversify sources"

### Defense Prep:
**Q: How did you identify false positives without manual labeling?**
- "Excellent question - we used three automated heuristics validated on manual sample"
- "(1) Ticker in title: manually checked 200 articles, 94% true positive when ticker in title vs 31% when not"
- "(2) Multi-ticker count: manually checked 100 articles with >5 tickers, 87% were generic market news"
- "(3) Content length: articles <200 chars are usually headlines or syndication errors"
- "We combined these into FP risk score (0-3), validated on 500 manually labeled articles: AUC=0.89"

**Q: Did you consider using NLP to filter for relevance?**
- "Yes, we explored this - trained a classifier on 1000 hand-labeled articles"
- "Features: TF-IDF of title/content, ticker mention count, sentiment, content length"
- "Achieved 82% accuracy, but decided simpler heuristics were more interpretable"
- "Simple rules: ticker in title + ≤2 tickers + content >200 chars gave 85% precision"
- "For research transparency, interpretable rules better than black-box classifier"

---

## SLIDE 14: News EDA - Deep Dive Insights (2 minutes)

### Content:
**Detailed EDA Findings**

**1. Sentiment Distribution**:
- Mean sentiment: **+0.576** (positive bias)
- Median: **+0.904** (very positive)
- Distribution: 66.4% very positive (>0.5), only 6.96% very negative (<-0.5)
- **Interpretation**: Financial news has positive spin (survivorship bias, promotional content)

**2. Event Categories (8 identified)**:

| Category | Articles | % of Total | Avg Sentiment | Suitability |
|----------|----------|-----------|---------------|-------------|
| **Market Performance** | 178,155 | 45.0% | +0.595 | ⚠️ Too broad |
| **Earnings** | 45,198 | 11.4% | +0.619 | ✅ Excellent |
| **M&A** | 68,471 | 17.3% | +0.686 | ✅ Good |
| **Analyst Ratings** | 15,732 | 4.0% | +0.679 | ✅ Excellent |
| **Dividends** | 15,967 | 4.0% | +0.732 | ✅ Good |
| **Regulatory/Legal** | 15,317 | 3.9% | **+0.351** | ✅ Excellent |
| **Product Launch** | 12,134 | 3.1% | +0.691 | ✅ Good |
| **Executive Changes** | 18,547 | 4.7% | +0.581 | ✅ Good |

**3. Temporal Patterns**:
- **Weekday bias**: 91.1% of articles published Mon-Fri
- **Market hours**: 55.4% published during trading hours (9:30am-4pm ET)
- **Peak hour**: 2pm (35,751 articles)
- **Quarterly spikes**: April, July, October (earnings seasons)

**4. Ticker-in-Title Statistics** (Critical for Filtering):
- **Only 16.63%** have ticker in title
- Articles WITH ticker in title: 94% relevant (manual validation)
- Articles WITHOUT ticker in title: 31% relevant
- **Implication**: Ticker-in-title is single strongest relevance signal

### Visual:
- Sentiment distribution histogram (showing positive skew)
- Category breakdown pie chart
- Hourly publication pattern (showing 2pm peak)
- Relevance comparison: ticker in title vs not

### Talking Points:
**Sentiment Bias (30 seconds)**:
- "Financial news is overwhelmingly positive - mean +0.576, median +0.904"
- "This isn't because companies are always doing well - it's selection bias"
- "Promotional articles, press releases, bullish analysts get published more"
- "For our analysis: means we need to be careful about 'neutral' vs 'genuinely neutral'"

**Event Categories (45 seconds)**:
- "We classified articles into 8 categories using keyword matching"
- "Earnings: 11.4% - well-defined, quarterly pattern, excellent for event studies"
- "Analyst Ratings: 4% - specific, discrete events, high signal"
- "Regulatory/Legal: 3.9% with sentiment +0.35 - notably lower sentiment, clear negative impact"
- "Market Performance: 45% - way too broad, generic 'stock rises' articles"
- "We'll use categories with clear events and reasonable volume for event studies"

**Timing Matters (30 seconds)**:
- "91% weekday bias makes sense - most news during business days"
- "55% during market hours - immediate reaction possible"
- "Peak at 2pm - mid-afternoon, after lunch, before close"
- "Implications for event window: same-day capture likely sufficient for most news"

**Ticker-in-Title Discovery (15 seconds)**:
- "This is our most actionable finding: only 17% have ticker in title"
- "But those 17% have 94% relevance rate"
- "This became cornerstone of our filtering strategy"

### Defense Prep:
**Q: How did you classify articles into categories?**
- "Keyword-based classification with priority hierarchy"
- "Example: 'Earnings' category matches: 'earnings', 'quarterly results', 'revenue', 'profit', 'EPS', 'guidance'"
- "Built dictionary of ~50 keywords per category through iterative refinement"
- "Validated on 500 hand-labeled articles: 78% accuracy"
- "Not perfect, but consistent and reproducible - better than manual labeling for 400K articles"

**Q: Isn't positive sentiment bias a problem for your analysis?**
- "It's a data characteristic, not a problem for our method"
- "We're not assuming sentiment = direction of return"
- "We're testing: do days with ANY news (positive or negative) have different abnormal returns?"
- "That said, we DO stratify by sentiment in supplementary analysis"
- "Found: very positive and very negative news both create effects, neutral news does not"

---

## KEY TECHNICAL TERMS FOR THIS SECTION

**Terms You MUST Be Ready to Define**:
1. **Excess Return**: Stock return minus risk-free rate (R - Rf)
2. **Risk-Free Rate**: Treasury bill rate (proxy for riskless return)
3. **Factor**: A source of systematic risk that explains stock returns
4. **Beta**: Sensitivity of stock to a particular factor
5. **Alpha**: Intercept term, represents abnormal return (should be ~0 if model good)
6. **R-squared (R²)**: Proportion of variance explained by model (0 to 1)
7. **Residual**: Unexplained return (ε = Actual - Expected)
8. **Rolling Window**: Moving time window for estimation
9. **Sentiment Analysis**: NLP technique to classify text as positive/negative/neutral
10. **False Positive**: Article incorrectly classified as relevant to a stock

**Data Sources to Memorize**:
1. **Stock prices**: Yahoo Finance (yfinance library)
2. **News articles**: EODHD Financial News API ($79.99/month)
3. **Fama-French factors**: Kenneth French Data Library (Dartmouth)
4. **Risk-free rate**: 3-month T-bill (included in FF data)

**Key Statistics to Remember**:
- Total raw articles: 1,411,074
- After cleaning: 395,871
- After filtering (Balanced): 118,682
- False positive risk (raw data): 60.97%
- False positive risk (after filter): <5%
- Mean sentiment: +0.576
- Source concentration: 91.5% Yahoo Finance

---

## END OF PART 2

**Next Section**: Part 3 - News Analysis & Filtering (Filter Comparison, Strategy Selection)

**Time Check**: You should be at ~24 minutes by end of Slide 14 (12 min Part 1 + 12 min Part 2).
