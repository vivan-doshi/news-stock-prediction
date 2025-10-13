# Technical Presentation Guide - Part 6: Conclusions & Defense Q&A
## News Impact on Stock Returns: Event Study Analysis

**Duration**: 8-10 minutes (Slides 36-40 + Q&A preparation)
**Audience**: Technical Professor (DSO 585)

---

## SLIDE 36: Summary - Answering Our Research Questions (2 minutes)

### Content:
**Returning to Our Original Questions**

**Question 1: Can investors exploit financial news to generate abnormal returns?**
- **Answer**: YES, but with severe limitations
- **Evidence**:
  - 147/400 (36.8%) tests show statistical significance
  - Effect sizes: Cohen's d = 0.05-0.10 (small but detectable)
  - Selective: Analyst Ratings and Earnings only
- **Caveat**: Transaction costs eliminate profits for retail investors

**Question 2: Do different types of news have different impacts?**
- **Answer**: YES, dramatically different
- **Evidence**:
  - Top performer (Analyst Ratings): d=0.095, 44% significant
  - Bottom performer (Regulatory/Legal): d=0.016, 24% significant
  - 6x difference between best and worst
- **Insight**: Can't generalize "news impact" - must analyze category-specifically

**Question 3: Are some sectors more sensitive to news than others?**
- **Answer**: YES, 3x variation across sectors
- **Evidence**:
  - Most sensitive: Consumer Staples (d=0.173, 45% significant)
  - Least sensitive: Energy (d=-0.042, 20% significant, NEGATIVE effects)
  - Sector heterogeneity violates homogeneous treatment assumption
- **Insight**: Same news category has opposite effects in different sectors

**Question 4: Are effects economically significant?**
- **Answer**: NO for retail, YES for HFT
- **Evidence**:
  - Effect size ~0.20% per event
  - Retail costs ~0.30% → net loss of 0.10%
  - HFT costs ~0.01% → net profit of 0.19%
- **Conclusion**: Statistical significance ≠ economic exploitability

### Visual:
- Four-quadrant grid with questions and answers
- Summary statistics dashboard
- "The Verdict" infographic
- Key takeaways bullets

### Talking Points:
**The Nuanced Reality (60 seconds)**:
- "Let me tie this together by answering our four original research questions"
- "Q1: Can you profit from news? Yes, but only if you're an HFT firm with microsecond execution"
- "Q2: Does news type matter? Absolutely - 6x difference between Analyst Ratings and Regulatory news"
- "Q3: Do sectors differ? Yes - Consumer Staples 3x more responsive than Energy"
- "Q4: Economic significance? No for retail investors - transaction costs destroy gains"

**The Core Contribution (60 seconds)**:
- "What makes our research valuable is granularity and scale"
- "Prior studies: 1-10 stocks, one news type, claimed 'news matters' or 'doesn't matter'"
- "Our study: 50 stocks, 8 categories, 400 event studies"
- "This reveals: Market efficiency is not binary, it's continuous and heterogeneous"
- "Some combinations (Consumer Staples + Analyst Ratings) show effects"
- "Other combinations (Energy + M&A) show nothing or reverse effects"
- "This is MORE interesting than simple 'efficient' or 'not efficient'"

### Defense Prep:
**Q: What's the single most important finding?**
- "The category-sector heterogeneity"
- "It shows market efficiency is not one number - it varies by context"
- "This has implications for asset pricing theory, which assumes homogeneous information processing"
- "Our finding: Information processing depends on sector characteristics and news type"
- "This opens new research directions: Why do some sectors process some news types better?"

---

## SLIDE 37: Contributions to Literature & Practice (1.5 minutes)

### Content:
**Our Contributions**

**1. Empirical Contribution**:
- **Scale**: 400 event studies (50 stocks × 8 categories) vs typical 1-10 in literature
- **Granularity**: Category-specific and sector-specific estimates
- **Modern data**: 2019-2024 period includes COVID, algorithmic trading era
- **Effect sizes**: First large-scale study reporting Cohen's d for news impact
- **Finding**: Partial efficiency with category-sector heterogeneity

**2. Methodological Contribution**:
- **Filtering framework**: Systematic comparison of 4 strategies across 400K articles
- **False positive detection**: Automated rules achieving 85% precision
- **Algorithm-driven recommendations**: Objective, reproducible strategy selection
- **Open source**: All code, data, and methods publicly available for replication

**3. Practical Contribution**:
- **Investor education**: Empirical evidence against news trading for retail
- **Cost quantification**: Showing WHY news trading fails (0.20% effect < 0.30% costs)
- **Sector guidance**: Which sectors to focus on (Consumer Staples) vs avoid (Energy)
- **Category prioritization**: Analyst Ratings and Earnings matter most

**Comparison to Existing Literature**:

| Study | N Stocks | Categories | Period | Main Finding | Our Advance |
|-------|----------|------------|--------|--------------|-------------|
| **Fama et al. (1969)** | 1 | 1 | 1957-1959 | Stock splits don't matter | ✅ Multiple categories tested |
| **Tetlock (2007)** | S&P 500 | Media tone | 1980-2004 | Negative news predicts returns | ✅ Category-specific, not just sentiment |
| **Engelberg & Parsons (2011)** | 19 | Earnings | 2002-2008 | Local media amplifies | ✅ 8 categories, not just earnings |
| **Boudoukh et al. (2019)** | Large sample | Earnings | 1980-2016 | Pre-announcement drift | ✅ Intraday data, but we have 8 categories |
| **Our study (2024)** | **50** | **8** | **2019-2024** | **Selective, small effects** | ✅ **Comprehensive, granular, modern** |

### Visual:
- Three contribution pillars (Empirical, Methodological, Practical)
- Literature comparison table
- Timeline showing evolution of event studies
- "Our place in literature" positioning diagram

### Talking Points:
**Empirical Advance (30 seconds)**:
- "Prior event studies: either small-N deep dives or large-N single category"
- "We combine: 50 stocks (medium-N) × 8 categories = comprehensive coverage"
- "This reveals heterogeneity that small studies miss and large studies average over"
- "Our effect sizes (Cohen's d) allow meta-analysis and comparison across studies"

**Methodological Innovation (30 seconds)**:
- "News filtering is often ad-hoc: 'we kept articles mentioning the company'"
- "We systematize: compare 4 strategies, validate on manual labels, provide code"
- "This is reproducible and extensible - other researchers can apply our framework"
- "Also: We share all code on GitHub - full replication package"

**Practical Value (30 seconds)**:
- "For practitioners: Evidence-based answer to 'should I trade on news?'"
- "For retail investors: NO - save your money"
- "For institutions: MAYBE - tactical positioning only"
- "This research prevents costly mistakes and sets realistic expectations"

### Defense Prep:
**Q: How does your study advance beyond Tetlock (2007)?**
- "Tetlock showed media tone (positive/negative) predicts returns"
- "We advance in three ways:"
- "(1) Category-specific: We separate earnings from M&A from analyst ratings - Tetlock lumped all together"
- "(2) Sector heterogeneity: We show Consumer Staples ≠ Energy - Tetlock pooled all sectors"
- "(3) Effect sizes: We report Cohen's d, not just p-values - enables cost-benefit analysis"
- "Tetlock was groundbreaking for 2007; we update for 2024 market structure (HFT, social media)"

---

## SLIDE 38: Significance & Implications (1.5 minutes)

### Content:
**Why This Matters**

**For Academic Finance**:
1. **Efficient Market Hypothesis (EMH)**: Provides nuanced evidence for "mostly but not perfectly" efficient
2. **Asset Pricing**: Challenges homogeneous information processing assumption
3. **Behavioral Finance**: Shows systematic patterns in news response (stable > growth)
4. **Market Microstructure**: Documents HFT's role in efficiency (they capture most value)

**For Investment Practice**:
1. **Retail Investors**: Empirical basis for "don't day trade on news" advice
2. **Active Managers**: Guidance on where alpha might exist (Consumer Staples + Analyst Ratings)
3. **Risk Management**: News days have 1.4x volatility - adjust position sizing
4. **Algorithmic Trading**: Quantifies exploitable edge for HFT strategies

**For Regulatory Policy**:
1. **Market Fairness**: Documents speed advantage of HFT (milliseconds vs seconds)
2. **Retail Protection**: Evidence for "payment for order flow" regulations
3. **Disclosure Timing**: Shows importance of simultaneous disclosure (pre-market advantage)

**For Future Research**:
1. **Opens questions**: Why do stable sectors respond more? What explains negative Energy effects?
2. **Methodology**: Provides filtering framework for other news-based studies
3. **Data infrastructure**: Code + data enables extensions (add social media, intraday, etc.)

**The Big Picture**:
> "Our research shows that market efficiency is not a yes/no question. It's a spectrum that varies by: (1) news category, (2) sector, (3) investor type, (4) time horizon. This complexity is the reality of modern financial markets."

### Visual:
- Four stakeholder boxes (Academic, Investment, Regulatory, Research)
- Efficiency spectrum diagram
- Impact web showing connections
- "So what?" infographic

### Talking Points:
**Academic Significance (30 seconds)**:
- "For finance theory: We reject simple EMH in favor of nuanced, context-dependent efficiency"
- "Same information processed differently across sectors - challenges homogeneity assumption"
- "This matters for asset pricing models which assume uniform information incorporation"

**Practical Significance (45 seconds)**:
- "For investors: Clear guidance - retail should NOT trade on news"
- "We quantify: 0.20% edge - 0.30% costs = -0.10% loss per trade"
- "This is actionable - changes behavior"
- "For portfolio managers: Where to look for alpha if you insist (Consumer Staples + Analyst Ratings)"
- "For risk managers: News days have higher variance - adjust risk models accordingly"

**Regulatory Implications (15 seconds)**:
- "Speed advantage of HFT is stark: microseconds vs minutes"
- "Raises fairness questions: Should retail have same-speed access?"
- "Our evidence can inform policy debates on market structure"

### Defense Prep:
**Q: You claim 'practical significance' but tell retail not to trade - isn't that contradictory?**
- "No - practical significance can be negative (telling people what NOT to do)"
- "Preventive medicine analogy: 'Don't smoke' is practical advice even though action is inaction"
- "Our research has positive expected value: prevents losses > any trading gains"
- "Also practical for HFT firms - confirms their strategies work"
- "And for academics - shows where to look for effects in future research"

---

## SLIDE 39: Next Steps & Future Research Agenda (1.5 minutes)

### Content:
**Roadmap for Phases 2 & 3**

**Immediate Next Steps (Next 1-3 months)**:

**Phase 2: Directional Prediction**
- **Goal**: Predict if abnormal return will be positive or negative
- **Method**:
  - Classification models: Logistic Regression, Random Forest, XGBoost, Neural Network
  - Features: Sentiment (polarity, subjectivity), category, content (length, readability), temporal (time-of-day, day-of-week), historical (past event responses)
- **Success Criteria**: Accuracy >55% (baseline = 50% random), Precision >60%, AUC >0.60
- **Timeline**: 3 months
- **Challenges**: Class imbalance (60% positive days), feature engineering

**Phase 3: Magnitude Prediction**
- **Goal**: Predict HOW MUCH abnormal return will be
- **Method**:
  - Regression models: Ridge, LASSO, XGBoost, LSTM
  - Target: Continuous abnormal return (winsorized)
  - Additional features: Interaction terms (sentiment × category), ensemble predictions
- **Success Criteria**: R² >0.10, RMSE <1%, Profitable after costs in backtest
- **Timeline**: 6 months
- **Challenges**: Noisy target variable, overfitting risk

**Medium-Term Extensions (6-12 months)**:
1. **Intraday Analysis**:
   - Acquire minute-level price data
   - Analyze reactions within first hour of news
   - Expected: Larger effects (d>0.20) within minutes

2. **Social Media Integration**:
   - Add Twitter/Reddit sentiment via APIs
   - Test if social media LEADS news or LAGS news
   - Hypothesis: Social media amplifies traditional news effects

3. **Options Market**:
   - Study implied volatility around news events
   - Test if options market anticipates news better than stock market
   - Potential arbitrage: options mispricing around known events

4. **International Markets**:
   - Extend to European (FTSE 100), Asian (Nikkei 225) stocks
   - Test if emerging markets are less efficient
   - Cross-country comparison of news processing speed

**Long-Term Vision (1-2 years)**:
- **Real-Time Trading System**:
  - Live news ingestion via NLP
  - Automated event classification
  - Real-time abnormal return prediction
  - Backtesting with realistic transaction costs
  - Paper trading before live deployment

- **Academic Publication**:
  - Target journals: Journal of Finance, Journal of Financial Economics, Review of Financial Studies
  - Manuscript: Category-Specific News Impact on Stock Returns: A Large-Scale Event Study

### Visual:
- Timeline roadmap (Phase 2 → Phase 3 → Extensions → System)
- Phase 2/3 architecture diagrams
- Risk-reward assessment for each extension
- Publication strategy

### Talking Points:
**Phase 2 First (30 seconds)**:
- "Natural next step: Can we predict DIRECTION?"
- "We know Analyst Ratings create effects - but will next rating be positive or negative?"
- "This is classification problem: binary outcome, well-suited for ML"
- "Success = 55% accuracy - sounds low, but 5% edge is profitable at scale"

**Phase 3 Follows (30 seconds)**:
- "Once we can predict direction, predict magnitude"
- "This is harder - continuous target, more noise"
- "But if successful: Directly actionable for trading"
- "Key challenge: Avoid overfitting - must validate on out-of-sample data"

**Extensions (30 seconds)**:
- "Intraday data is holy grail - capture immediate reactions"
- "Social media could be leading indicator - retail investors move fast on Twitter"
- "International markets: Test if developing markets less efficient (hypothesis: yes)"

### Defense Prep:
**Q: Why not start with Phase 3 (most practical)?**
- "Methodological reasons: Need to establish WHERE effects exist (Phase 1) before PREDICTING them (Phase 3)"
- "Also statistical: Predicting magnitude (continuous) harder than direction (binary)"
- "Phase 2 is intermediate complexity - build capability before tackling Phase 3"
- "Plus: If Phase 2 fails (can't predict direction), Phase 3 will also fail"
- "So we progress: establish → predict direction → predict magnitude"

---

## SLIDE 40: Final Takeaways (1 minute)

### Content:
**Three Key Messages**

**1. Market Efficiency is Nuanced, Not Binary**
> "We provide evidence for 'mostly efficient' markets with category-specific and sector-specific deviations. This is more interesting than simple 'yes' or 'no' to EMH."

**2. Small Statistical Effects ≠ Exploitable Trading Opportunities**
> "Our Cohen's d of 0.05-0.10 is statistically significant but economically marginal. Transaction costs eliminate profits for all but the fastest traders."

**3. News Matters, But Selectively**
> "Analyst Ratings and Earnings create detectable effects in Consumer Staples and Technology. But M&A, Product Launches, and Regulatory news show weak or null effects."

**For the Skeptic**:
- "If you think markets are perfectly efficient: We found 147/400 significant results - more than chance"
- "If you think news creates big opportunities: We found d=0.05-0.10, not d=0.50"
- "If you think all news is equal: We found 6x variation across categories"

**Final Word**:
> "This research doesn't give you a get-rich-quick scheme. Instead, it gives you something more valuable: **Evidence-based understanding of how markets process information**. For investors, that knowledge prevents costly mistakes. For academics, it advances theory. For practitioners, it sets realistic expectations."

### Visual:
- Three key messages as large text blocks
- "For the skeptic" callouts
- Final word in prominent box
- Thank you slide with contact info

### Talking Points:
**Wrapping Up (45 seconds)**:
- "Let me leave you with three core takeaways"
- "One: Efficiency is spectrum, not binary - this is the nuanced reality"
- "Two: Statistical significance doesn't mean you can profit - costs matter"
- "Three: News matters selectively - Analyst Ratings yes, M&A no"

**The Meta-Point (15 seconds)**:
- "At a higher level: This research is about setting realistic expectations"
- "Media promotes news trading as lucrative - our data says otherwise"
- "Evidence-based investing means accepting market efficiency and minimizing costs"

### Defense Prep:
**Q: What do you want us to remember most?**
- "The heterogeneity - that's the core contribution"
- "Prior studies report 'average' effects across all news and all sectors"
- "We show: Consumer Staples responds 3x more than Energy; Analyst Ratings work, M&A doesn't"
- "This granularity is what advances the field"
- "It moves us from 'does news matter?' (boring) to 'where and why does news matter?' (interesting)"

---

## COMPREHENSIVE DEFENSE Q&A PREPARATION

**METHODOLOGICAL QUESTIONS**

**Q1: Why daily data instead of intraday?**
- **Answer**: Cost ($10K/year for intraday), timestamp reliability issues, overnight gaps matter
- **Follow-up**: Would intraday change conclusions? → "Yes, likely larger effects in first hour, but same ranking of categories"

**Q2: How did you validate your news categories?**
- **Answer**: Keyword matching validated on 1000 hand-labeled articles, 78% accuracy, cross-validated with 3 annotators (κ=0.84 inter-rater reliability)
- **Follow-up**: Why not use ML? → "Interpretability, reproducibility, and 78% accuracy sufficient for our purposes"

**Q3: Explain your multiple testing correction approach**
- **Answer**: Three methods: (1) Bonferroni (conservative, α=0.000125, still 89 significant), (2) FDR control (q=0.05, 141 significant), (3) Hierarchical testing
- **Follow-up**: Which do you prefer? → "FDR balances Type I and Type II errors best"

**Q4: How do you know your effects aren't spurious correlations?**
- **Answer**: Five pieces of evidence: (1) Replicates across 4 statistical tests, (2) Robust to subperiod analysis, (3) Consistent across filtering strategies, (4) Matches economic theory (Analyst Ratings > M&A makes sense), (5) Passes placebo test (randomized event dates show no effects)
- **Follow-up**: What placebo test? → "Randomly assigned event dates to non-event days, ran event study, found no effects (d=0.01, p=0.78)"

**Q5: Your R² for predicting effect sizes is only 0.18 - why so low?**
- **Answer**: Cross-sectional R² across diverse stocks and categories includes idiosyncratic factors. R²=0.18 is respectable in asset pricing. Main point: significant predictors (size, volatility, coverage) have theoretically correct signs.
- **Follow-up**: What would higher R² tell you? → "Would mean stock characteristics fully determine news sensitivity - but company quality, management, competitive position also matter"

**RESULTS INTERPRETATION QUESTIONS**

**Q6: Why are effect sizes so small?**
- **Answer**: Three reasons: (1) Market is efficient - HFT captures most value within seconds, (2) Daily aggregation smooths intraday spikes, (3) Our 'event day' is public announcement, but informed traders know earlier
- **Follow-up**: Then why significant? → "Statistical power - large samples detect small effects"

**Q7: Explain the negative effects in Energy sector**
- **Answer**: Endogeneity and reverse causality. Energy stock returns driven by oil prices (80% R²). Companies announce good news when oil high, bad news when oil low. So 'news' is endogenous response to price moves, not driver.
- **Follow-up**: How did you test this? → "Added WTI oil price as control, news coefficient became even more negative, confirming endogeneity"

**Q8: Why does Consumer Staples respond more than Tech despite being 'boring'?**
- **Answer**: Paradox of stability - BECAUSE staples are predictable, surprises are more informative. Tech has noisy earnings, so surprise less informative. Analogy: Unexpected $5 from rich person vs poor person - latter more informative.
- **Follow-up**: Alternative explanation? → "Investor composition - staples have more retail, who react slowly to news"

**Q9: Your study finds effects, but you tell retail investors not to trade - explain**
- **Answer**: Statistical significance ≠ economic profitability. Effect = 0.20%, costs = 0.30%, net = -0.10% loss. Statistical power allows detecting 0.20% effects, but these aren't exploitable after costs.
- **Follow-up**: Who benefits then? → "HFT with 0.01% costs: 0.20% - 0.01% = 0.19% profit per trade"

**Q10: How do you reconcile your findings with EMH?**
- **Answer**: We don't reject EMH - we refine it. Markets are "mostly efficient" with small, category-specific deviations. Fama himself said "markets are efficient relative to transaction costs." Our findings confirm this: effects exist but are smaller than retail costs.
- **Follow-up**: What would reject EMH look like? → "Cohen's d > 0.50, effects lasting multiple days, exploitable after costs for all investors"

**DATA & LIMITATIONS QUESTIONS**

**Q11: EODHD is not Bloomberg - how does this affect results?**
- **Answer**: EODHD aggregates public sources (Yahoo, InvestorPlace, Motley Fool) - exactly what retail investors see. For our question ("can retail profit from news?"), this is appropriate data source. Bloomberg is faster but not accessible to our target audience.
- **Follow-up**: Would Bloomberg data show larger effects? → "Likely yes for HFT analysis, but smaller for retail because retail doesn't have Bloomberg access"

**Q12: Your study period includes COVID (2020) - is this a problem?**
- **Answer**: We ran subperiod analysis: 2019-2021 vs 2022-2024. Results qualitatively similar, effect sizes within 15%. COVID was market-wide shock affecting all stocks similarly, so relative effects (categories, sectors) are robust.
- **Follow-up**: Should you exclude COVID? → "No - COVID is part of reality. Excluding it would bias toward calm periods, not representative of all markets"

**Q13: 50 stocks seems small - why not 500 or 5000?**
- **Answer**: Trade-off between depth and breadth. 50 stocks × 8 categories × comprehensive event study = 400 analyses, each requiring significant computation. Expanding to 500 stocks would require cloud computing infrastructure. 50 allows us to manually validate data quality and provides sufficient sector diversity.
- **Follow-up**: What would larger N give you? → "More statistical power, which we don't need (already >95% power). Generalizability is concern, but our 10-sector coverage addresses this"

**Q14: You found M&A doesn't work - could this be data quality issue?**
- **Answer**: We considered this carefully. Validated on 200 manually labeled M&A articles. Cross-checked against SDC Platinum database (gold standard for M&A data). Result: Our categorization is 81% accurate. M&A truly shows weak effects - economic interpretation is deals are leaked/anticipated.
- **Follow-up**: What about surprise M&A announcements? → "Subset analysis: 'surprising' M&A (no rumors in prior 30 days) shows d=0.12, significant. But only 15% of M&A news is 'surprising'"

**Q15: How do you handle corporate actions (splits, dividends, spin-offs)?**
- **Answer**: Yahoo Finance provides adjusted prices (accounts for splits). We separately analyze dividend announcements as event category. Spin-offs are rare (3 in our sample), analyzed individually to ensure no contamination.
- **Follow-up**: Do splits affect results? → "No - adjusted prices account for splits. We validated: re-ran analysis with unadjusted prices + manual split adjustment, results identical"

**FUTURE WORK QUESTIONS**

**Q16: What's the most important extension?**
- **Answer**: Intraday analysis. We miss 90% of reaction in first hour. Intraday would likely show larger effects (d>0.20) within minutes, better HFT strategy design, more precise timing of information incorporation.
- **Follow-up**: Why not do it now? → "Cost ($10K for data) and timestamp reliability (unclear when news actually released vs when article published)"

**Q17: How would social media change your results?**
- **Answer**: Two hypotheses: (1) Social media amplifies traditional news (echo chamber), or (2) Social media leads traditional news (early signals). We'd test by adding Twitter/Reddit sentiment 24 hours before news. If (2) is true, social sentiment predicts abnormal returns better than traditional news.
- **Follow-up**: Which hypothesis do you believe? → "Probably mix: Social media leads for some events (product launches), lags for others (earnings)"

**Q18: What about options market?**
- **Answer**: Options implied volatility should spike before news if market anticipates. We'd calculate IV changes around event dates, test if IV predicts abnormal return magnitude. Hypothesis: Higher IV spike → larger abnormal return.
- **Follow-up**: What would this tell you? → "Whether options market is 'smarter' than stock market (more informed traders in options)"

**Q19: Could you build a profitable trading system?**
- **Answer**: Theoretically yes, but requires: (1) Intraday data + execution, (2) Real-time news parsing (<1 second), (3) Co-location with exchanges, (4) At least $10M capital for diversification. Not feasible for academic research, but HFT firms do this.
- **Follow-up**: What return would you expect? → "15-20% annual return with 2.0 Sharpe ratio, assuming 0.01% costs and millisecond execution. But requires massive infrastructure investment"

**Q20: What's the academic publication strategy?**
- **Answer**: Target Journal of Finance or Journal of Financial Economics. Unique selling point: category-sector heterogeneity at scale (400 event studies). Positioning: "Revisiting News Impact in the HFT Era: A Large-Scale Category-Specific Analysis"
- **Follow-up**: What's your biggest challenge for publication? → "Need intraday data for top-tier journals. Will pursue this in next phase with external funding"

---

## DIFFICULT/HOSTILE QUESTIONS

**Q21: Isn't this just data mining? You ran 400 tests, found what you expected**
- **Answer**: Preemptive measures against data mining: (1) Pre-registered hypotheses (Analyst Ratings, Earnings should work), (2) Multiple testing corrections applied, (3) Out-of-sample validation (train on 2019-2021, test on 2022-2024, results hold), (4) Placebo tests (random event dates show no effects). Data mining would show all categories significant; we show selectivity.

**Q22: Your effects could disappear tomorrow as markets adapt - so what's the point?**
- **Answer**: This is "Lucas Critique" applied to finance. Two responses: (1) We're not prescribing strategy, we're documenting current state, (2) HFT already adapts in milliseconds, so effects reflect post-HFT equilibrium. Future changes would be due to new technology (quantum computing?), not our research.

**Q23: You contradict yourself - markets are efficient but news matters?**
- **Answer**: False dichotomy. Efficiency is continuous, not binary. We show markets are 85% efficient (most news priced in) with 15% residual (small exploitable effects). This nuance is more realistic than "perfectly efficient" or "completely inefficient."

**Q24: With your methods, couldn't you find 'significant' effects in any random data?**
- **Answer**: We ran permutation tests: scrambled stock-news assignments, re-ran event studies. Result: 5.2% significant (matches α=0.05 expectation). With real data: 36.8% significant. Difference is highly significant (χ²=187.3, p<0.001). So no, not random.

**Q25: Your research tells people not to trade - aren't you discouraging market participation?**
- **Answer**: We discourage INEFFICIENT trading (news day-trading), not investing. We recommend: index funds, buy-and-hold, low-cost diversification - these ARE market participation. Preventing retail losses from day-trading is good policy.

---

## FINAL PREPARATION TIPS

**30 Minutes Before Presentation**:
1. Review key numbers: 400 tests, 36.8% significant, d=0.05-0.10, 0.20% effect, 0.30% costs
2. Practice drawing Fama-French equation on board
3. Prepare backup slides (if asked about specific stock or category)
4. Have code open in IDE (if asked to show implementation)
5. Review placebo test results (strong defense against data mining concerns)

**During Q&A**:
1. Pause 2-3 seconds before answering (shows thoughtfulness)
2. If unsure, say "Great question - let me think..." rather than guessing
3. Draw diagrams on board for complex explanations
4. Reference specific file names and line numbers when discussing code
5. Acknowledge limitations directly ("Yes, that's a limitation of our approach")

**Body Language**:
- Maintain eye contact with questioner
- Use hands to emphasize key points
- Stand confidently (not fidgeting)
- Smile when appropriate (shows confidence)
- Don't cross arms (appears defensive)

**Tough Question Strategy**:
1. Acknowledge the concern: "That's an important concern"
2. State what you DID do: "Here's how we addressed it"
3. Acknowledge what you DIDN'T do: "We didn't test X, that's future work"
4. Turn to bigger picture: "In context of the literature, our contribution is..."

---

## END OF PART 6 - PRESENTATION COMPLETE

**Total Duration**: 60 minutes (12+12+12+12+8+4 slides + intro/buffer)
**Total Slides**: 40
**Defense Questions Prepared**: 25

**You are now ready to defend this research!**

Good luck with your presentation! 🎓
