# News Filtering Strategies Explained

This document explains the 4 news filtering strategies implemented in [25_comprehensive_news_filter.py](02-scripts/25_comprehensive_news_filter.py).

## Overview

The filtering system addresses three major challenges in news-based financial analysis:
1. **False positives** - Articles mentioning a ticker but not being about that company
2. **Multi-ticker articles** - Market-wide news that dilutes company-specific signals
3. **Duplicates** - Same story published multiple times

---

## 1. Precision Strategy

**Goal:** Minimize false positives, maximize article quality

**Why this name?** Like precision in statistics (true positives / predicted positives), this strategy is very strict to ensure every article retained is genuinely relevant.

### Filtering Rules
- ✅ Ticker MUST be in title
- ✅ ≤2 tickers mentioned (avoids market-wide news)
- ✅ Content ≥200 characters (substantive articles only)
- ✅ Only high-impact categories: Earnings, Product Launch, Regulatory/Legal, Analyst Ratings
- ✅ Removes duplicates
- ✅ **One article per day** with highest sentiment magnitude

### Use Case
Clean event studies where you want zero noise and highest quality signals.

### Code Reference
See [25_comprehensive_news_filter.py:155-173](02-scripts/25_comprehensive_news_filter.py#L155)

---

## 2. Recall Strategy

**Goal:** Comprehensive coverage, don't miss any potentially relevant news

**Why this name?** Like recall in statistics (true positives / actual positives), this strategy casts a wide net to capture all possible relevant articles, even at the cost of including some noise.

### Filtering Rules
- ✅ Ticker in title OR extreme sentiment (|polarity| > 0.7)
- ✅ ≤5 tickers mentioned (more permissive)
- ✅ Content ≥100 characters (lower threshold)
- ✅ Must match at least one event category
- ✅ No per-day limits

### Use Case
When you can't afford to miss important events, even if you get some false positives. Good for exploratory analysis.

### Code Reference
See [25_comprehensive_news_filter.py:175-187](02-scripts/25_comprehensive_news_filter.py#L175)

---

## 3. Balanced Strategy ⭐ **RECOMMENDED**

**Goal:** Optimal trade-off between precision and recall

**Why this name?** Balances the strictness of Precision with the comprehensiveness of Recall.

### Filtering Rules
- ✅ Ticker in title OR (≤2 tickers AND extreme sentiment |polarity| > 0.6)
- ✅ Content ≥200 characters
- ✅ Includes 6 major categories:
  - Earnings
  - Product Launch
  - Regulatory/Legal
  - Analyst Ratings
  - Executive Changes
  - Dividends
- ✅ Deduplicates within stock-date pairs

### Use Case
Most research scenarios - gives you good coverage without too much noise. This is the recommended starting point for most analyses.

### Code Reference
See [25_comprehensive_news_filter.py:189-207](02-scripts/25_comprehensive_news_filter.py#L189)

---

## 4. Category-Specific Strategy

**Goal:** Apply different filtering rules for different event types

**Why this name?** Recognizes that different news categories have different characteristics and should be filtered differently.

### Filtering Rules by Category

#### Earnings
- Ticker in title required
- ≤3 tickers allowed
- Content ≥200 characters

#### Product Launch / Regulatory
- Ticker in title OR extreme sentiment (|polarity| > 0.7)
- ≤2 tickers allowed
- Content ≥300 characters (higher threshold for substance)

#### Analyst Ratings
- Ticker in title required (most strict)
- EXACTLY 1 ticker only
- Content ≥150 characters

#### Executive Changes / Dividends
- Ticker in title required
- ≤2 tickers allowed
- Content ≥200 characters

### Use Case
Research focused on specific event types or when you want category-tailored filtering. Best for robust academic research where different events may require different treatment.

### Code Reference
See [25_comprehensive_news_filter.py:209-258](02-scripts/25_comprehensive_news_filter.py#L209)

---

## Event Categories

All strategies use these 8 event categories:

1. **Earnings** - Quarterly results, revenue, EPS, profit announcements
2. **Product Launch** - New products, releases, unveilings
3. **Executive Changes** - CEO, leadership, board appointments/resignations
4. **M&A** - Mergers, acquisitions, takeovers, joint ventures
5. **Regulatory/Legal** - SEC actions, lawsuits, investigations, settlements
6. **Analyst Ratings** - Upgrades, downgrades, price targets
7. **Dividends** - Dividend announcements, payout changes
8. **Market Performance** - Stock price movements, trading activity

---

## False Positive Detection

The system calculates a **False Positive (FP) Score** from 0 (best) to 3 (worst):

| Indicator | Points |
|-----------|--------|
| Ticker NOT in title | +1 |
| More than 2 tickers mentioned | +1 |
| Content < 200 characters | +1 |

**FP Score = 0**: High confidence the article is genuinely about this stock
**FP Score = 3**: Likely a false positive

---

## Comparison Metrics

The system generates these comparative statistics:

- **Retention Rate** - % of original articles retained
- **Average FP Score** - Quality indicator (lower is better)
- **Ticker in Title %** - Relevance indicator (higher is better)
- **Articles per Stock** - Coverage distribution
- **Category Distribution** - Event type breakdown
- **Sentiment Statistics** - Mean and extremity measures

---

## Usage Example

```python
from pathlib import Path
import sys
sys.path.append('02-scripts')

# Import the filtering system
from 25_comprehensive_news_filter import ComprehensiveNewsFilter

# Process a single stock
filter_tool = ComprehensiveNewsFilter(
    ticker='AAPL',
    sector='Technology',
    company_name='Apple Inc.'
)

# Run analysis
results = filter_tool.process()

# Access filtered datasets
precision_df = results['Precision']['df']
balanced_df = results['Balanced']['df']
```

---

## Output Files

For each stock and strategy combination, the system generates:

1. **Filtered dataset**: `{TICKER}_{strategy}_filtered.csv`
   - Contains all filtered articles with metadata

2. **Event dates**: `{TICKER}_{strategy}_event_dates.csv`
   - Just the dates for event study analysis

3. **Comparison summary**: `filtering_comparison_summary.csv`
   - Side-by-side statistics for all strategies

4. **Visualizations**: `filtering_strategies_comparison.png`
   - 8-panel dashboard comparing all strategies

5. **Recommendations**: `strategy_recommendations.csv`
   - Stock-specific strategy recommendations

---

## Naming Philosophy

The strategy names reflect **information retrieval concepts**:

- **Precision** = Quality over quantity (minimize false positives)
- **Recall** = Quantity to capture everything (minimize false negatives)
- **Balanced** = The sweet spot between the two
- **Category-Specific** = Adaptive filtering based on event characteristics

This mirrors the precision-recall tradeoff in machine learning and information retrieval, making the naming intuitive for researchers familiar with these concepts.

---

## Recommendations by Use Case

| Use Case | Recommended Strategy | Reason |
|----------|---------------------|--------|
| Event study research | **Precision** | Clean, high-quality events with minimal noise |
| Exploratory analysis | **Recall** | Don't miss anything potentially interesting |
| General modeling | **Balanced** | Good coverage without excessive false positives |
| Multi-event studies | **Category-Specific** | Different events need different treatment |
| Publication-ready research | **Category-Specific** | Most rigorous, defensible methodology |

---

## Technical Details

**Language**: Python 3.x
**Dependencies**: pandas, numpy, matplotlib, seaborn
**Input**: EODHD news CSV files (`{TICKER}_eodhd_news.csv`)
**Output Directory**: `03-output/news_filtering_comparison/`

**Processing time**: ~2-3 minutes for 50 stocks with ~200K total articles

---

*Last updated: 2025-10-13*