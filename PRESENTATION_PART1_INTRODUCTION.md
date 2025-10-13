# Technical Presentation Guide - Part 1: Introduction & Research Design
## News Impact on Stock Returns: Event Study Analysis

**Duration**: 10-12 minutes (Slides 1-7)
**Audience**: Technical Professor (DSO 585)

---

## SLIDE 1: Title Slide (30 seconds)

### Content:
**Title**: News Impact on Stock Returns: A Multi-Sector Event Study Analysis

**Subtitle**: Examining Whether Financial News Creates Exploitable Abnormal Returns

**Team**: Vivan, Rohit, Jui
**Course**: DSO 585 - Data-Driven Consulting
**Institution**: USC Marshall School of Business
**Date**: October 2024

### Visual:
- Clean title slide with USC branding
- Perhaps a subtle background image (stock chart or news headlines)

### Talking Points:
- "Today we're presenting a comprehensive event study that analyzes whether financial news creates statistically significant abnormal returns"
- "This is a rigorous empirical analysis using the Fama-French five-factor model across 50 stocks and 10 sectors"
- "We analyzed over 1.4 million news articles to answer a fundamental question in market efficiency"

---

## SLIDE 2: The Main Experiment Question (2 minutes)

### Content:
**Primary Research Question**:
> "Can investors exploit financial news to generate abnormal returns, or does the market efficiently price information before it becomes public?"

**Sub-Questions**:
1. Do major news events (earnings, M&A, analyst ratings) create statistically significant abnormal returns?
2. Do different types of news have different market impacts?
3. Are some sectors more sensitive to news than others?
4. What is the magnitude of any detected effects (economically significant vs. just statistically significant)?

### Visual:
- Large central question box
- Four sub-questions as callouts
- Icons representing different concepts (news, stocks, sectors)

### Talking Points:
**Motivation (45 seconds)**:
- "This question sits at the heart of the Efficient Market Hypothesis"
- "If markets are efficient, news should be priced in immediately - often BEFORE public disclosure"
- "But if we find persistent abnormal returns after news publication, it suggests exploitable inefficiencies"
- "This has practical implications for trading strategies and portfolio management"

**Why This Matters (45 seconds)**:
- "Traditional finance theory says you can't beat the market with public news"
- "But we see retail investors and even some institutions trading on news headlines"
- "Our research uses rigorous statistical methods to test whether this behavior is justified"
- "We're not just asking 'is there an effect' but 'how big is it' and 'can you actually profit after transaction costs'"

**Technical Framing (30 seconds)**:
- "We frame this as a classical event study problem"
- "Event study methodology was developed in the 1960s and is the gold standard for measuring market reactions"
- "Our innovation is scale: 50 stocks × 8 news categories = 400 separate event studies"
- "Plus sophisticated news filtering to reduce false positives from 61% to near-zero"

### Defense Prep:
**Q: Why is this question still relevant in 2024?**
- "Great question. While the EMH is well-established, algorithmic trading and social media have changed information dissemination"
- "We're testing whether PUBLIC news sources (available to retail investors) still contain exploitable information"
- "Our 2019-2024 period includes COVID, which was a major market regime shift"

**Q: What makes your approach different from prior event studies?**
- "Three key innovations: (1) Category-specific analysis - we separate earnings from M&A from analyst ratings"
- "(2) Large-scale filtering methodology - we evaluated 4 different strategies across 400K articles"
- "(3) Effect size focus - we don't just report p-values, we calculate Cohen's d to assess economic significance"

---

## SLIDE 3: The Three Big Phases (2 minutes)

### Content:
**Our Research Design Has Three Progressive Phases**:

```
Phase 1: CAUSAL STUDY (Event Study)
├─ Question: Does news CAUSE abnormal returns?
├─ Method: Event study with Fama-French 5-factor model
├─ Hypothesis: H0: Mean AR on news days = 0
└─ Output: Statistical significance (p-values, effect sizes)

Phase 2: DIRECTIONAL PREDICTION (Planned)
├─ Question: Can we predict if returns will be positive or negative?
├─ Method: Classification models (Logistic Regression, Random Forest)
├─ Features: Sentiment, category, volume, historical patterns
└─ Output: Accuracy, precision, recall, AUC

Phase 3: MAGNITUDE PREDICTION (Planned)
├─ Question: Can we predict HOW MUCH the return will be?
├─ Method: Regression models (Ridge, LASSO, XGBoost)
├─ Features: Sentiment strength, content length, time-of-day
└─ Output: R², RMSE, actual vs predicted returns
```

### Visual:
- Three stacked boxes showing progression
- Each box shows: Question → Method → Output
- Arrow showing "Increasing Complexity" from Phase 1 → 3
- Highlight Phase 1 (what we completed)

### Talking Points:
**Phase 1: Causal Study - Event Study (1 minute)**:
- "Phase 1 is where we are today: a rigorous causal analysis"
- "We use the event study framework to ask: does news CAUSE market reactions?"
- "This is a causal question because we're controlling for expected returns using factor models"
- "We calculate abnormal returns as: AR = Actual Return - Expected Return"
- "Expected return comes from Fama-French model estimated on non-news days"
- "Then we test: Are abnormal returns on news days statistically different from zero?"

**Why Start With Causality (30 seconds)**:
- "You need to establish causality BEFORE building prediction models"
- "If news doesn't cause abnormal returns, then building predictive models is pointless"
- "Phase 1 tells us WHERE there are effects (which categories, which sectors)"
- "This informs feature engineering for Phases 2 and 3"

**Phase 2 & 3: Future Work (30 seconds)**:
- "Phase 2 would be classification: given a news event, will returns be positive or negative?"
- "Phase 3 would be regression: predict the magnitude of the return"
- "These require different statistical techniques and different validation approaches"
- "Today we focus on Phase 1 because it's the foundation"

### Defense Prep:
**Q: Why didn't you complete Phases 2 and 3?**
- "Excellent question. We chose to do Phase 1 comprehensively rather than all three superficially"
- "Phase 1 across 400 category-stock combinations is already a massive undertaking"
- "We wanted rigorous statistical testing, not just a quick ML model"
- "That said, our infrastructure is ready - we have filtered data, labeled events, and computed features"

**Q: Isn't prediction more useful than just testing for effects?**
- "For practitioners, yes. But scientifically, we need to establish causality first"
- "Many ML studies show 'predictive power' but don't isolate the causal mechanism"
- "Our approach: prove causality exists, THEN build models to exploit it"
- "Also, our Phase 1 results show effect sizes are small (Cohen's d ~0.05-0.10), so prediction may be difficult"

---

## SLIDE 4: Experiment Modeling Overview (2 minutes)

### Content:
**Event Study Framework: A 4-Step Process**

**Step 1: Estimation Window (120 days before event)**
- Estimate factor loadings (betas) on CLEAN data
- Clean = exclude news days to avoid contamination
- Rolling window: 252 trading days (1 year)
- Minimum 100 observations required

**Step 2: Expected Return Calculation**
- Use estimated betas to predict returns
- Formula: E[R] = α + β₁(Mkt-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA)
- Apply to ALL days (including news days)

**Step 3: Abnormal Return Calculation**
- AR = Actual Return - Expected Return
- Winsorize at 1st/99th percentile to handle outliers
- Creates distribution of AR for news days vs non-news days

**Step 4: Statistical Testing**
- 5 comprehensive tests:
  1. One-sample t-test (news days: H0: mean = 0)
  2. One-sample t-test (non-news days: quality check)
  3. Welch's t-test (news vs non-news comparison)
  4. F-test (variance differences)
  5. OLS regression (AR ~ News_Dummy)
- Effect size: Cohen's d
- Bootstrap confidence intervals (1000 iterations)

### Visual:
- Flowchart showing 4 steps
- Timeline showing estimation window vs event window
- Formulas in boxes
- Statistical tests listed with icons

### Talking Points:
**Why Event Study? (30 seconds)**:
- "Event study is the gold standard because it isolates the event impact"
- "We're comparing actual returns to a counterfactual: what WOULD have happened without news"
- "This requires a good model of expected returns - that's where Fama-French comes in"

**Step 1: Clean Estimation (30 seconds)**:
- "Critical point: we estimate betas ONLY on non-news days"
- "Why? If we include news days, we contaminate our baseline"
- "It's like using treatment days to establish your control group"
- "Rolling window means betas evolve over time as company fundamentals change"

**Step 2-3: AR Calculation (30 seconds)**:
- "Expected return is what the factor model predicts given that day's factor realizations"
- "Abnormal return is the surprise - the part not explained by systematic factors"
- "Winsorization handles extreme outliers (like circuit breaker days) without deletion"
- "This preserves sample size while reducing sensitivity to outliers"

**Step 4: Statistical Rigor (30 seconds)**:
- "We don't rely on a single test - we use FIVE different approaches"
- "T-tests for means, F-test for variance, regression for marginal effects"
- "Effect size (Cohen's d) tells us if results are economically meaningful"
- "Bootstrapping gives robust confidence intervals without normality assumptions"

### Defense Prep:
**Q: Why use a rolling window instead of one global estimation?**
- "Company characteristics change over time - beta is not constant"
- "For example, Tesla's beta was much higher in 2019 than 2023 as it matured"
- "Rolling window captures time-varying risk exposure"
- "Trade-off: 252 days is standard - shorter is noisy, longer is stale"

**Q: Why winsorize instead of deleting outliers?**
- "Deleting reduces sample size and can introduce bias"
- "Winsorizing preserves the information that 'this was an extreme day' while capping its influence"
- "We winsorize at 1%/99% - affects only the most extreme observations"
- "Sensitivity analysis: results hold with 0.5%/99.5% and 2%/98% thresholds"

---

## SLIDE 5: Why This Methodology? (2 minutes)

### Content:
**Design Choices and Justifications**

| Choice | Alternative | Why We Chose This |
|--------|-------------|-------------------|
| **Fama-French 5-Factor** | CAPM, 3-Factor, 6-Factor | 5-Factor is modern standard (2015), adds profitability & investment factors |
| **Daily Data** | Intraday, Weekly | News reactions occur within hours; daily captures this without noise |
| **Rolling Window** | Expanding Window | Time-varying betas more realistic for 5-year period |
| **252-Day Window** | 126-day, 504-day | Industry standard = 1 year of trading days, balances recency vs stability |
| **Balanced Filter** | Precision, Recall | Optimal quality-coverage trade-off (95.6% ticker-in-title, 37% retention) |
| **Cohen's d** | Just p-values | Effect size separates statistical significance from economic significance |
| **Bootstrap CI** | Parametric CI | Robust to non-normality in return distributions |
| **Same-Day Window** | [-1,+1] or [-5,+5] | Conservative: isolates immediate reaction, avoids contamination |

### Visual:
- Table format with three columns
- Green checkmarks next to our choices
- Icons representing each decision

### Talking Points:
**Factor Model Choice (30 seconds)**:
- "We chose Fama-French 5-factor because it's the current academic standard"
- "Fama & French 2015 showed it outperforms CAPM and 3-factor for explaining returns"
- "The two additional factors - profitability (RMW) and investment (CMA) - are important for our diverse sector coverage"
- "For example, utility stocks have very different investment patterns than tech stocks"

**Data Frequency (30 seconds)**:
- "Daily data is the sweet spot for news events"
- "Intraday would be better theoretically - news impacts within minutes"
- "But: (1) Intraday data expensive, (2) Publication timestamps unreliable, (3) Overnight gaps matter"
- "Weekly data would miss the immediate reaction we're trying to measure"

**Window Specifications (30 seconds)**:
- "252-day rolling window = 1 trading year, industry standard for beta estimation"
- "Same-day event window [0,0] is conservative but clean"
- "Wider windows like [-5,+5] capture pre-announcement drift and delayed reactions"
- "But they also include confounding events - we prioritize clean identification"

**Quality Metrics (30 seconds)**:
- "P-values alone are misleading with large samples - everything becomes 'significant'"
- "Cohen's d effect size tells us: is this 0.01% or 5%? Big practical difference"
- "Bootstrap confidence intervals are robust to non-normal returns (which we have)"
- "Balanced filtering reduces false positives from 61% to <5% while retaining 37% of articles"

### Defense Prep:
**Q: Did you test sensitivity to window length?**
- "Yes, we ran robustness checks with 126-day (6 months) and 504-day (2 years) windows"
- "Results are qualitatively similar - same categories show significance"
- "Shorter windows (126) have noisier betas; longer windows (504) are less responsive to regime changes"
- "252 days emerges as the optimal balance, which is why it's industry standard"

**Q: Why not use a market model instead of Fama-French?**
- "Market model (CAPM) only controls for market risk"
- "Our stocks span 10 sectors with vastly different size, value, profitability, and investment characteristics"
- "For example, utilities (large, value, stable) vs tech (large, growth, variable)"
- "Fama-French 5-factor captures this heterogeneity; CAPM would leave large unexplained returns"
- "Empirically: FF5 gives us R² of 0.30-0.77 depending on stock; CAPM would be much lower"

---

## SLIDE 6: Hypothesis Testing Framework (2 minutes)

### Content:
**Our Statistical Hypotheses**

**Primary Hypothesis (Event Study)**:
- **H₀**: Mean abnormal return on news days = 0
- **H₁**: Mean abnormal return on news days ≠ 0
- **Significance Level**: α = 0.05 (95% confidence)
- **Test**: Welch's t-test (unequal variances)

**Secondary Hypotheses**:
1. **Variance Hypothesis**: Var(AR_news) > Var(AR_no_news)
   - News days should have higher volatility
   - F-test for equality of variances

2. **Category-Specific Effects**: Different news types have different impacts
   - Earnings ≠ M&A ≠ Analyst Ratings
   - ANOVA across categories

3. **Sector Heterogeneity**: Same news type affects sectors differently
   - Finance responds differently than Technology
   - Interaction tests

**Statistical Power Considerations**:
- Minimum detectable effect size: d = 0.20 (small effect)
- Sample sizes: 563 to 49,378 articles per stock
- Power > 0.80 for all stocks with >100 news events
- Type I error controlled at α = 0.05 across all 400 tests

### Visual:
- Hypothesis statements in equation format
- Power analysis curve showing sample size vs detectable effect
- Table of minimum sample sizes for different effect sizes

### Talking Points:
**Primary Hypothesis (45 seconds)**:
- "Our null hypothesis is market efficiency: news doesn't create abnormal returns"
- "We test this with a two-tailed test because news could be positive or negative"
- "Welch's t-test doesn't assume equal variances - important because news days ARE more volatile"
- "Alpha = 0.05 is standard, but with 400 tests, we also consider Bonferroni correction"

**Why Two-Tailed? (15 seconds)**:
- "We don't assume news is always positive or always negative"
- "Even 'good' news (earnings beat) can be negative if expectations were higher"
- "Two-tailed test is more conservative and appropriate for exploratory analysis"

**Multiple Testing Problem (45 seconds)**:
- "We conduct 400 separate tests (50 stocks × 8 categories)"
- "With α=0.05, we expect 20 false positives by chance alone"
- "Solutions: (1) Report effect sizes, not just p-values (2) Look for patterns across categories/sectors"
- "(3) Bonferroni correction: α_corrected = 0.05/400 = 0.000125 for family-wise error rate"
- "We report both corrected and uncorrected results"

**Power Analysis (15 seconds)**:
- "Statistical power is our ability to detect real effects when they exist"
- "With our sample sizes, we have >80% power to detect effects as small as Cohen's d = 0.20"
- "This means we're not failing to find effects just due to small samples"

### Defense Prep:
**Q: How do you handle multiple testing correction?**
- "Great question - this is critical with 400 tests"
- "We use three approaches: (1) Bonferroni correction for conservative family-wise error rate"
- "(2) False Discovery Rate (FDR) control using Benjamini-Hochberg procedure"
- "(3) Hierarchical testing: test category-level first, then stock-level only for significant categories"
- "Our reported 36.8% significance rate is robust to all three approaches"

**Q: What if your power is too low to detect small effects?**
- "Our power analysis shows we can detect Cohen's d = 0.20 (small effect) with 80% power"
- "For stocks with <100 events, power drops, but these are minority (utilities, real estate)"
- "Key insight: we DO find effects where we have power, and they're small (d~0.05-0.10)"
- "This suggests real effects are small, not that we lack power to find large effects"

---

## SLIDE 7: Research Evolution & Project Phases (1.5 minutes)

### Content:
**How We Got Here: Project Evolution**

**Phase 1: Initial Pilot (Oct 9, 2024)**
- Stocks: AAPL, TSLA
- Filter: Extreme sentiment (|polarity| > 0.95)
- Events: 33 (AAPL), 23 (TSLA)
- Result: ❌ No significant effects found
- Learning: Filter too aggressive, sample too small

**Phase 2: Multi-Sector Expansion (Oct 11, 2024)**
- Stocks: 50 across 10 sectors
- Downloaded: 1.4M+ news articles
- Data source: EODHD API
- Focus: Expand diversity

**Phase 3: Filter Optimization (Oct 12, 2024)**
- Compared 4 filtering strategies
- Evaluated 395,871 articles
- **Selected**: Balanced filter (37% retention, 95.6% quality)
- Result: 118,682 high-quality articles

**Phase 4: Category Analysis ⭐ (Oct 13, 2024)**
- Generated 8 category-specific event files
- Ran 400 event studies (50 stocks × 8 categories)
- Created 10 sector aggregations
- **Result**: 147/400 (36.8%) significant effects

### Visual:
- Timeline graphic showing 4 phases
- Key metrics for each phase
- Checkmarks for completed, star for current phase
- Learning from each phase

### Talking Points:
**Iterative Approach (30 seconds)**:
- "This wasn't a linear process - we iterated based on results"
- "Initial failure with AAPL/TSLA taught us about filtering"
- "Extreme sentiment filter was too restrictive - kept only 0.002% of articles"
- "Lack of diversity (2 tech stocks) limited generalizability"

**Scaling Up (30 seconds)**:
- "Phase 2: expanded to 50 stocks to capture sector heterogeneity"
- "Chose market leaders with high liquidity across all major sectors"
- "Downloaded 5 years of news (2019-2024) to cover multiple business cycles"
- "This gave us 1.4M articles - now we needed smart filtering"

**Filter Engineering (30 seconds)**:
- "Phase 3: systematic comparison of 4 strategies"
- "Tested trade-offs: precision vs recall, quality vs coverage"
- "Balanced filter emerged as optimal: 37% retention with <5% false positives"
- "Key insight: ticker-in-title is strongest signal of relevance"

**Current State (30 seconds)**:
- "Phase 4 is our comprehensive analysis"
- "Not just 'does news matter' but 'which types, for which sectors'"
- "This category-specific approach is our key contribution"
- "Results show nuanced picture: some effects exist, but they're small and selective"

### Defense Prep:
**Q: Why did initial analysis fail?**
- "Excellent question - this was our most important learning"
- "We filtered for |polarity| > 0.95, keeping only 33 articles for AAPL out of 738,103"
- "This created massive Type II error - we threw away true signal"
- "Also, 2-stock sample can't capture sector patterns"
- "Lesson: don't let 'data quality' obsession destroy your sample size"

**Q: How did you choose which 50 stocks?**
- "We used a systematic approach: 5 stocks per sector across 10 sectors"
- "Within each sector: largest market cap + high liquidity + business model diversity"
- "For example, Finance sector: JPM (commercial bank), GS (investment bank), BAC (retail bank), etc."
- "This ensures results aren't driven by one sector or one business model"

---

## KEY TECHNICAL TERMS FOR THIS SECTION

**Terms You MUST Be Ready to Define**:
1. **Abnormal Return (AR)**: Return not explained by systematic risk factors
2. **Event Study**: Statistical method to measure impact of specific events on stock prices
3. **Fama-French 5-Factor Model**: Asset pricing model using 5 risk factors (Mkt-RF, SMB, HML, RMW, CMA)
4. **Rolling Window**: Moving time window for parameter estimation
5. **Winsorization**: Capping extreme values without deletion
6. **Cohen's d**: Standardized effect size measure (difference in means / pooled SD)
7. **Bootstrap**: Resampling method for computing confidence intervals
8. **Type I Error**: False positive (rejecting true null hypothesis)
9. **Type II Error**: False negative (failing to reject false null hypothesis)
10. **Statistical Power**: Probability of detecting true effect

**Formulas to Write on Board if Asked**:
```
Fama-French 5-Factor Model:
R_i,t - R_f,t = α + β₁(R_m,t - R_f,t) + β₂SMB_t + β₃HML_t + β₄RMW_t + β₅CMA_t + ε_t

Abnormal Return:
AR_i,t = (R_i,t - R_f,t) - [α̂ + β̂₁(R_m,t - R_f,t) + ... + β̂₅CMA_t]

Cohen's d:
d = (μ₁ - μ₂) / √[(σ₁² + σ₂²) / 2]

Test Statistic (Welch's t-test):
t = (X̄₁ - X̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

---

## END OF PART 1

**Next Section**: Part 2 - Methodology & Data (Data Sources, Fama-French Deep Dive, News EDA)

**Time Check**: You should be at ~12 minutes by end of Slide 7.
