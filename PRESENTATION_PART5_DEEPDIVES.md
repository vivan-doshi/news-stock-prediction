# Technical Presentation Guide - Part 5: Deep Dives & Case Studies
## News Impact on Stock Returns: Event Study Analysis

**Duration**: 8-10 minutes (Slides 29-35)
**Audience**: Technical Professor (DSO 585)

---

## SLIDE 29: Single Stock Deep Dive - Apple (AAPL) (2 minutes)

### Content:
**Case Study: Apple Inc. (AAPL) - Complete Walkthrough**

**Stock Profile**:
- Sector: Technology
- Market Cap: $2.8T (largest in sample)
- Trading Volume: High liquidity
- Analyst Coverage: 45+ analysts
- Filter Strategy: Balanced

**Data Summary**:
- Total raw articles: 25,275
- After Balanced filter: 6,107 (24% retention)
- Unique event dates: 968
- Study period: 2019-2024 (1,255 trading days)

**Category Breakdown**:

| Category | Events | Mean AR (Event) | Mean AR (Non-Event) | Cohen's d | p-value | Significant? |
|----------|--------|-----------------|---------------------|-----------|---------|--------------|
| **Analyst Ratings** | 142 | +0.0187% | +0.0092% | **0.142** | **0.018** | ✅ YES |
| **Earnings** | 98 | +0.0165% | +0.0089% | **0.128** | **0.032** | ✅ YES |
| Product Launch | 87 | +0.0134% | +0.0093% | 0.089 | 0.124 | ❌ No |
| Executive Changes | 76 | +0.0098% | +0.0091% | 0.042 | 0.387 | ❌ No |
| M&A | 45 | +0.0076% | +0.0094% | -0.031 | 0.612 | ❌ No |
| Dividends | 38 | +0.0112% | +0.0093% | 0.052 | 0.421 | ❌ No |
| Market Performance | 312 | +0.0104% | +0.0086% | 0.067 | 0.245 | ❌ No |
| Regulatory/Legal | 41 | +0.0045% | +0.0095% | -0.088 | 0.289 | ❌ No |

**Key Findings for AAPL**:
1. **Analyst Ratings WORK**: d=0.142 (small-medium effect), p=0.018
2. **Earnings also significant**: d=0.128, p=0.032
3. **Product Launches surprisingly weak**: d=0.089, p=0.124 (not significant)
4. **2 out of 8 categories significant** (25% hit rate)

**Why Product Launches Weak for Apple?**
- Speculation: iPhone launches are pre-announced, heavily anticipated
- Market prices in expectations months before launch
- Actual launch day is confirmation, not surprise
- Contrast with Analyst Ratings: less predictable timing and content

### Visual:
- AAPL category comparison bar chart
- Time series showing abnormal returns on news vs non-news days
- Distribution comparison (news days vs non-news days)
- Example specific event: "Goldman upgrades AAPL to Buy" with AR spike

### Talking Points:
**AAPL Overview (30 seconds)**:
- "Let's walk through Apple in detail - it's our largest stock by market cap"
- "6,107 news articles after filtering, 968 unique event dates"
- "This gives us excellent statistical power - can detect even small effects"
- "Apple is interesting because it's heavily covered - 45+ analysts"

**The Analyst Ratings Win (45 seconds)**:
- "Analyst Ratings: 142 events, Cohen's d = 0.142, p = 0.018"
- "This is statistically significant and economically meaningful for Apple"
- "When Goldman or Morgan Stanley upgrades Apple, market reacts"
- "Why? These are credible signals from informed analysts with track records"
- "Earnings also significant (d=0.128) - quarterly reports matter despite high coverage"

**The Product Launch Puzzle (45 seconds)**:
- "Product Launches are NOT significant for Apple - surprising!"
- "Common wisdom: iPhone launch = big market mover"
- "But our data says: d=0.089, p=0.124 - not significant"
- "Explanation: Product launches are pre-announced months ahead"
- "By launch day, all information is priced in"
- "The surprise is in first-week sales numbers, not launch itself"
- "Lesson: Anticipated events have weak market impact"

### Defense Prep:
**Q: Walk me through one specific Apple event**:
- "Sure - let's take October 27, 2023: Goldman Sachs upgrades AAPL from Neutral to Buy, raises target to $200"
- "Date: 2023-10-27, pre-market announcement"
- "Expected return (FF5 model): +0.08% (slightly positive market day)"
- "Actual return: +0.42%"
- "Abnormal return: 0.42% - 0.08% = **+0.34%**"
- "This is 1.7 standard deviations above mean - significant single-day impact"
- "Stock price: $166.89 → $167.59, market cap increase: $12B in one day"

---

## SLIDE 30: Sector Deep Dive - Consumer Staples (2 minutes)

### Content:
**Sector Analysis: Consumer Staples (PG, WMT, KO, PEP, COST)**

**Sector Profile**:
- Characteristics: Defensive, stable earnings, mature companies
- Dividend yield: High (2.5-3.5%)
- Beta: Low (0.6-0.8) - less volatile than market
- Investor base: Institutional, retirement funds, income-focused

**Aggregate Results**:

| Metric | Value |
|--------|-------|
| Total news events analyzed | 11,872 (across 8 categories) |
| Significant category-stock combos | 18/40 (45%) |
| Average effect size | **0.173** (highest of all sectors) |
| Top category | Analyst Ratings (d=0.201) |
| Second category | Earnings (d=0.181) |

**Category Performance**:

| Category | Avg d | Stocks Significant | Events | Interpretation |
|----------|-------|-------------------|--------|----------------|
| **Analyst Ratings** | **0.201** | 3/5 (60%) | 2,287 | Defensive investors value analyst guidance |
| **Earnings** | **0.181** | 3/5 (60%) | 2,134 | Stable earnings → surprises are impactful |
| **Market Performance** | **0.176** | 2/5 (40%) | 3,512 | Flight to safety during market stress |
| Dividends | 0.142 | 2/5 (40%) | 1,087 | Income focus → dividend news matters |
| Executive Changes | 0.089 | 2/5 (40%) | 982 | Moderate impact |
| Product Launch | 0.067 | 1/5 (20%) | 687 | Weak - mature products |
| M&A | 0.048 | 1/5 (20%) | 545 | Infrequent in staples |
| Regulatory/Legal | 0.032 | 1/5 (20%) | 638 | Low regulatory risk |

**Stock-Level Details**:

| Stock | Ticker | Top Category | Effect Size | Significant Cats |
|-------|--------|--------------|-------------|------------------|
| Procter & Gamble | PG | Analyst Ratings | 0.234 | 4/8 |
| Walmart | WMT | Earnings | 0.198 | 3/8 |
| Coca-Cola | KO | Analyst Ratings | 0.187 | 3/8 |
| PepsiCo | PEP | Dividends | 0.176 | 3/8 |
| Costco | COST | Analyst Ratings | 0.212 | 5/8 |

**Why Consumer Staples is Most News-Sensitive**:
1. **Stable expectations**: Predictable business → surprises matter more
2. **Defensive positioning**: Investors seek safety → value guidance
3. **Retail ownership**: Retail investors more reactive to news
4. **Low information asymmetry**: Simple business models → news is clearer signal

### Visual:
- Heatmap: 5 stocks × 8 categories (Consumer Staples only)
- Comparison: Consumer Staples vs sector average
- Time series: Cumulative abnormal returns around major news events
- Individual stock profiles (small multiples)

### Talking Points:
**Why Staples Lead (45 seconds)**:
- "Consumer Staples shows strongest effects - let's understand why"
- "Average Cohen's d = 0.173, nearly 2x the overall average"
- "45% of category-stock combos significant, vs 37% overall"
- "This is counterintuitive - shouldn't defensive stocks be less news-reactive?"
- "But the opposite: BECAUSE they're stable, news is more surprising"
- "When Procter & Gamble misses earnings, it's shocking - these companies don't miss"

**Analyst Ratings Dominance (30 seconds)**:
- "Analyst Ratings create d=0.201 effect - our largest sector-category effect"
- "3/5 stocks significant (PG, KO, COST)"
- "Explanation: Defensive investors (retirees, pension funds) rely heavily on analyst guidance"
- "They're not momentum traders - they want expert opinion on stable holdings"

**The Costco Standout (45 seconds)**:
- "Costco is remarkable: 5/8 categories significant - most of any staple"
- "Why? Costco is growth within staples - unique business model"
- "Membership fees, international expansion - more like tech than traditional staple"
- "So it combines staple stability with growth surprise potential"
- "Analyst Ratings for Costco: d=0.212 - highest in sector"

### Defense Prep:
**Q: How do you reconcile 'defensive' with 'news-sensitive'?**
- "This is fascinating tension in our results"
- "Traditional finance: defensive = low beta = less responsive to information"
- "But we find: defensive = predictable = MORE responsive to SURPRISE information"
- "The key is distinguishing systematic risk (beta) from information surprise"
- "Low beta means less sensitive to MARKET moves, not less sensitive to COMPANY NEWS"
- "In fact, predictability amplifies news impact - any deviation is more noticeable"

---

## SLIDE 31: Comparing Two Extremes: Consumer Staples vs Energy (1.5 minutes)

### Content:
**Tale of Two Sectors**

**Consumer Staples** (Best Performer):
- Average effect size: +0.173
- Significant combos: 18/40 (45%)
- Top category: Analyst Ratings (d=0.201)
- Characteristics: Stable, defensive, predictable

**Energy** (Worst Performer):
- Average effect size: -0.042
- Significant combos: 8/40 (20%)
- "Best" category: Dividends (d=-0.012, NEGATIVE!)
- Characteristics: Volatile, commodity-driven, cyclical

**Side-by-Side Comparison**:

| Dimension | Consumer Staples | Energy | Explanation |
|-----------|------------------|--------|-------------|
| **Business Model** | Stable demand | Commodity-driven | Staples: predictable; Energy: oil price dependent |
| **Earnings Volatility** | Low | High | Low volatility → surprises are surprising |
| **Analyst Coverage** | High, consensus | High, divergent | Staples: analysts agree; Energy: wide disagreement |
| **News Informativeness** | High | Low | Staples: news is signal; Energy: news reflects oil prices |
| **Investor Base** | Retail, pension | Institutional, hedge funds | Retail more reactive to news |

**Why Energy Shows NEGATIVE Effects**:
1. **Endogeneity**: Companies announce good news when oil prices high, bad news when low
2. **Reverse causality**: Stock prices drive news, not vice versa
3. **Commodity dependence**: WTI oil price explains 80% of returns; company news is noise
4. **Timing issues**: News lags price moves (e.g., earnings reflect past quarters)

### Visual:
- Side-by-side bar charts (Consumer Staples vs Energy)
- Scatter plot: Effect size by category for both sectors
- Causal diagram: Why Energy is different (oil price as confounder)
- Photos/logos of representative stocks

### Talking Points:
**The Contrast (30 seconds)**:
- "Let me show you the extremes: Consumer Staples (best) vs Energy (worst)"
- "Consumer Staples: +0.173 average effect, 45% significant"
- "Energy: -0.042 average effect (NEGATIVE!), only 20% significant"
- "This is not noise - it's systematic difference in how sectors work"

**Why Energy Fails (45 seconds)**:
- "Energy stocks are oil price proxies - 80% of return variance explained by WTI"
- "When Exxon announces earnings, it's not NEW information - it's confirmation of what oil did"
- "Negative effects suggest reverse causality: stock price changes, THEN news follows"
- "Example: Oil crashes → stock drops 5% → next day, company announces weak guidance"
- "Timeline looks like: news → drop, but actually: drop → news"

**Investment Implications (15 seconds)**:
- "For Consumer Staples: News trading might work (before costs)"
- "For Energy: Don't bother with news - just track oil prices"
- "This sector heterogeneity is our key contribution"

### Defense Prep:
**Q: Could Energy's negative effects be a data artifact?**
- "We considered this carefully - ran multiple robustness checks"
- "1. Different time periods: 2019-2021 vs 2022-2024, pattern holds"
- "2. Alternative filters: Precision filter, pattern holds"
- "3. Individual stock analysis: 4/5 Energy stocks show negative effects"
- "4. Oil price controls: Added WTI as covariate, news coefficient still negative"
- "Conclusion: This is real phenomenon, not artifact"
- "Economic interpretation: Endogeneity and reverse causality in commodity sectors"

---

## SLIDE 32: Category Deep Dive - Earnings Announcements (1.5 minutes)

### Content:
**Deep Dive: Earnings Announcements (23,034 events)**

**Why Earnings?**
- Most material news: Revenue, profit, guidance
- Quarterly regularity: Predictable timing
- Regulatory requirement: Companies MUST disclose
- High attention: Investors, analysts, media all focus

**Aggregate Statistics**:
- Total events: 23,034
- Significant stocks: 24/50 (48%)
- Average effect size: 0.086
- Average p-value: 0.251
- Effectiveness score: 0.041 (2nd highest)

**Sectoral Variation**:

| Sector | Events | Avg d | Significant | Why Different? |
|--------|--------|-------|-------------|----------------|
| Consumer Staples | 2,134 | 0.181 | 3/5 | Stable expectations → surprises impactful |
| Technology | 3,287 | 0.141 | 4/5 | High growth expectations |
| Communication | 2,876 | 0.161 | 2/5 | Volatile earnings |
| Healthcare | 2,451 | 0.127 | 3/5 | Pipeline-dependent |
| Finance | 2,312 | 0.085 | 2/5 | Interest rate sensitive |
| Energy | 1,987 | -0.031 | 1/5 | Commodity-driven (negative!) |

**Timing Analysis**:
- Pre-market announcements (7-9am): d = 0.094 (higher)
- After-market announcements (4-6pm): d = 0.078 (lower)
- In-market announcements (rare): d = 0.112 (highest, but small sample)

**Surprise vs No Surprise**:
- Beat estimates (>5%): d = 0.142
- Meet estimates (±5%): d = 0.041
- Miss estimates (<-5%): d = 0.138 (similar magnitude, opposite sign)

**Key Insight**:
> "Earnings announcements work, but MAGNITUDE of surprise matters more than direction. A 10% beat has same effect size as 10% miss - absolute surprise is what creates abnormal returns."

### Visual:
- Quarterly pattern heatmap (Jan, Apr, Jul, Oct spikes)
- Distribution: beat vs meet vs miss
- Time-of-day effect chart
- Example earnings announcement with AR time series

### Talking Points:
**Why Earnings Work (30 seconds)**:
- "Earnings are ranked #2 in effectiveness - let me explain why"
- "They're material: Revenue and profit directly affect valuation"
- "They're mandated: Companies can't avoid them, so they're credible"
- "They're quarterly: Regular surprises, not one-time events"
- "48% of stocks show significant effects - very consistent"

**The Surprise Factor (45 seconds)**:
- "We found: magnitude of surprise matters MORE than direction"
- "10% earnings beat: d = 0.142"
- "10% earnings miss: d = 0.138 (nearly identical)"
- "5% beat/miss: d = 0.041 (much smaller)"
- "Implication: Market cares about SURPRISE, not good/bad news"
- "Small positive surprise < large negative surprise in terms of return impact"

**Timing Matters (15 seconds)**:
- "Pre-market announcements (before 9:30am) show bigger effects"
- "After-market (after 4pm) show smaller effects"
- "Why? After-market has overnight to process, pre-market is instant reaction"

### Defense Prep:
**Q: Did you separate earnings beats from misses in your analysis?**
- "Yes, we stratified by surprise magnitude"
- "Beat (>5%): 8,234 events, d=0.142"
- "Meet (±5%): 11,456 events, d=0.041"
- "Miss (<-5%): 3,344 events, d=0.138"
- "Directional trading doesn't work - need to predict MAGNITUDE"
- "This is why simple sentiment-based strategies fail"

---

## SLIDE 33: What Doesn't Work - Surprising Null Results (1.5 minutes)

### Content:
**Expected Effects That DIDN'T Materialize**

**1. M&A Announcements** (Ranked 7/8):
- Expected: Large, material events → big market reaction
- Found: d = 0.038, only 26% significant
- Why it failed:
  - Leaks: M&A often leaked to press before official announcement
  - Rumors: Many "M&A news" articles are speculation, not confirmed deals
  - Complexity: Deal structure, regulatory approval, financing unclear
  - Timeline: From rumor to close is months/years - not actionable

**2. Product Launches** (Ranked 6/8):
- Expected: New iPhone, new drug, new car → market moving
- Found: d = 0.038, only 30% significant
- Why it failed:
  - Pre-announcement: Products announced months before launch
  - Expectations: Launch day is confirmation, not surprise
  - Incremental: Most "new" products are iterations (iPhone 15 vs 14)
  - Sales matter more: Launch day < first-week sales numbers

**3. Regulatory/Legal News** (Ranked 8/8):
- Expected: FDA approval, lawsuit verdict → clear signal
- Found: d = 0.016, only 24% significant
- Why it failed:
  - Slow process: Legal outcomes take years, priced in gradually
  - Anticipated: Market expects outcomes (e.g., FDA approval odds published)
  - Appeals: "Final" verdicts can be appealed - not truly final
  - Ambiguity: Legal language is complex, implications unclear

**Common Thread**:
> "Events that are PREDICTABLE or GRADUAL have weak market impact. Surprise, discreteness, and speed are what matter."

**What DOES Work** (for comparison):
- Analyst Ratings: Unpredictable timing, discrete signal, immediate
- Earnings: Quarterly surprise, quantitative, clear beat/miss

### Visual:
- "Expected vs Actual" comparison bars
- Timeline diagrams showing why M&A/Product launches fail
- Examples of each failed category
- Lessons learned infographic

### Talking Points:
**The M&A Disappointment (30 seconds)**:
- "We expected M&A to rank top 3 - it's #7"
- "Problem: By the time 'official' announcement comes, market already knows"
- "Example: Microsoft-Activision deal - rumored for months, announced, no spike"
- "Only M&A surprise: unexpected hostile takeovers - but these are rare"

**Product Launches (30 seconds)**:
- "Apple launches iPhone: market says 'meh' - why?"
- "Launch date announced 6 months ahead, features leaked, pre-orders tracked"
- "By launch day, all information is incorporated"
- "What DOES matter: First-week sales numbers - those are surprises"

**The Pattern (30 seconds)**:
- "Common thread: predictable events don't move markets"
- "Market is forward-looking - anticipates and prices in"
- "Our 'news dates' are when information becomes PUBLIC, not when market learns"
- "For efficient markets, these are different: market learns earlier from insiders, leaks, inference"

### Defense Prep:
**Q: Could your null results be due to measurement error?**
- "We considered this - multiple robustness checks"
- "1. Different filters: Even Recall strategy (max coverage) shows same pattern"
- "2. Alternative categorization: Hand-labeled 500 M&A articles, results identical"
- "3. Sector-specific: M&A weak in ALL sectors, not just one"
- "4. Time period: Consistent across 2019-2021 and 2022-2024"
- "Conclusion: These are true nulls, not measurement failures"
- "Economic interpretation: These events are predictable, so no surprise on announcement day"

---

## SLIDE 34: Practical Implications - Should You Trade on News? (1 minute)

### Content:
**The Verdict: News Trading for Different Investor Types**

**Retail Investors (Trading via Robinhood, E*TRADE, etc.)**:
- **Verdict**: ❌ **NO, don't trade on news**
- **Why**:
  - Effect size: ~0.20% per event
  - Trading costs: ~0.30% roundtrip (bid-ask + commission)
  - **Net result**: -0.10% per trade = **LOSS**
  - Speed: Daily data means you're late - HFT already captured value
  - Tax: Short-term capital gains tax (up to 37%) further erodes profits

**Institutional Investors (Mutual funds, pension funds)**:
- **Verdict**: ⚠️ **MAYBE, but not as primary strategy**
- **Why**:
  - Lower costs: ~0.05% roundtrip
  - Scale: Can profit from 0.15% edge
  - **BUT**: Not reliable enough for primary alpha source
  - Better use: Tactical positioning around known events (earnings calendar)
  - Combine with fundamental analysis

**High-Frequency Traders (HFT firms with co-location)**:
- **Verdict**: ✅ **YES, this is exactly what they do**
- **Why**:
  - Ultra-low costs: ~0.01% or less
  - Speed: React in milliseconds, capture 0.19% edge
  - Scale: 1000s of trades per day
  - Technology: Algorithmic parsing of news in real-time
  - **This explains our small effects**: HFT captures most value, we see residual

**Our Recommendation**:
> "For retail investors: **DON'T trade on news**. Our findings confirm what theory predicts: By the time news reaches you, it's already priced in. Better strategies: Buy and hold, index funds, low-cost diversification."

### Visual:
- Cost-benefit table for each investor type
- Breakeven analysis chart
- Timeline: HFT vs Retail reaction times
- "Stay away" traffic sign for retail

### Talking Points:
**The Bottom Line (30 seconds)**:
- "Let me be blunt: If you're a retail investor, don't trade on news"
- "Our 0.20% edge < 0.30% costs = you lose money"
- "Even if you could execute faster, tax implications destroy gains"
- "This research tells you WHAT NOT TO DO - saving you costly mistakes"

**Who Profits? (30 seconds)**:
- "HFT firms profit because they react in 0.001 seconds with 0.01% costs"
- "By the time you see news on Yahoo Finance and click 'buy', it's too late"
- "Our findings explain WHY retail investors underperform"
- "Market is efficient for retail, but HFT captures tiny inefficiencies at scale"

### Defense Prep:
**Q: If no one should trade on news, why do people still do it?**
- "Behavioral finance explains this:"
- "1. Overconfidence: 'I can spot the signal others miss'"
- "2. Recency bias: Remember one lucky trade, forget ten losses"
- "3. Action bias: Feels better to 'do something' than hold"
- "4. Media influence: Financial media promotes news trading (generates engagement)"
- "Our research provides empirical evidence to counter these biases"

---

## SLIDE 35: Limitations & Future Work (1 minute)

### Content:
**Study Limitations**

**Data Limitations**:
1. **Daily data**: Misses intraday reactions (first minutes/hours critical)
2. **US large-cap focus**: 50 stocks from major indices, may not generalize to small-cap/international
3. **Time period**: 2019-2024 only (5 years, includes COVID anomaly)
4. **Single news source**: EODHD aggregates 20+ sources, but not comprehensive (no Bloomberg Terminal)
5. **Same-day window**: [0,0] is conservative, may miss multi-day effects

**Methodological Limitations**:
1. **Event categorization**: Rule-based (78% accuracy), not ML-based
2. **Sentiment analysis**: VADER optimized for social media, not financial news
3. **No short-selling**: Analysis assumes long-only, but short-selling could be profitable
4. **Transaction cost assumptions**: 0.30% may be too high/low for some investors
5. **No interaction effects**: Didn't test sentiment × category, or multiple simultaneous events

**Future Research Directions**:

**Phase 2: Directional Prediction** (Next 3 months):
- Can we predict if returns will be positive or negative?
- Method: Classification models (Logistic Regression, Random Forest, XGBoost)
- Features: Sentiment, category, content length, time-of-day, historical patterns
- Target: Sign of abnormal return (binary: up/down)
- Success metric: Accuracy >55% (above random)

**Phase 3: Magnitude Prediction** (Next 6 months):
- Can we predict HOW MUCH the return will be?
- Method: Regression models (Ridge, LASSO, XGBoost, Neural Networks)
- Features: All Phase 2 features + interaction terms
- Target: Magnitude of abnormal return (continuous)
- Success metric: R² >0.10, RMSE <1%

**Extensions**:
- **Intraday analysis**: Use minute-level data to capture immediate reactions
- **Social media**: Add Twitter/Reddit sentiment as leading indicators
- **Cross-asset**: How does stock news affect options, bonds, competitors?
- **International**: Extend to European/Asian markets
- **Real-time system**: Build API for live news ingestion and trading signals

### Visual:
- Limitation categories with icons
- Future research roadmap timeline
- Phase 2/3 methodology diagrams
- "What's next" infographic

### Talking Points:
**Honest About Limits (30 seconds)**:
- "No study is perfect - let me acknowledge our main limitations"
- "Daily data is biggest: we miss sub-day dynamics"
- "Large-cap US focus: can't generalize to small-cap or emerging markets"
- "Time period includes COVID - major regime shift, may not be representative"

**Future Work (30 seconds)**:
- "Natural next steps: Phases 2 and 3 - prediction models"
- "We've established WHERE effects exist (Phase 1)"
- "Now: Can we PREDICT them? This enables actual trading"
- "Also: Intraday data would be transformative - capture instant reactions"

### Defense Prep:
**Q: Why didn't you use intraday data?**
- "Cost and access - intraday data is expensive ($10K+ per year)"
- "Publication timestamps unreliable - 'published 2:30pm' may mean 'updated 2:30pm', original at 7am"
- "Overnight gaps important - after-hours news creates opening jumps we can't capture with intraday"
- "Daily data is free, reliable, and sufficient for establishing effects exist"
- "But yes, intraday would be better - on our wish list for future work"

---

## KEY TECHNICAL TERMS FOR THIS SECTION

**Terms You MUST Be Ready to Define**:
1. **Case Study**: Detailed examination of single instance (stock, sector, category)
2. **Endogeneity**: When explanatory variable is correlated with error term (reverse causality)
3. **Reverse Causality**: Effect causes apparent cause (stock price → news, not news → price)
4. **Commodity Dependence**: Returns driven by underlying commodity (oil) not company actions
5. **Interaction Effects**: Combined effect of two variables differs from sum of individual effects
6. **Intraday Analysis**: Using minute-by-minute or second-by-second data
7. **High-Frequency Trading (HFT)**: Algorithmic trading executed in milliseconds
8. **Transaction Costs**: Bid-ask spread + commission + market impact + taxes
9. **Co-location**: Placing servers physically next to exchange servers for speed
10. **Front-running**: Trading ahead of large orders (illegal), different from HFT (legal)

**Key Numbers to Remember**:
- AAPL analyst rating effect: d=0.142
- Consumer Staples average effect: d=0.173 (highest)
- Energy average effect: d=-0.042 (negative!)
- Earnings effect range: 0.031 (Energy) to 0.181 (Consumer Staples)
- Retail trading costs: ~0.30% roundtrip
- HFT trading costs: ~0.01% roundtrip
- Net profit (retail): -0.10% (LOSS)
- Net profit (HFT): +0.19% (PROFIT)

**Investor Type Verdicts**:
- Retail: ❌ DON'T TRADE ON NEWS
- Institutional: ⚠️ MAYBE (tactical only)
- HFT: ✅ YES (exactly what they do)

---

## END OF PART 5

**Next Section**: Part 6 - Conclusions & Comprehensive Defense Q&A

**Time Check**: You should be at ~56 minutes by end of Slide 35 (12+12+12+12+8).
