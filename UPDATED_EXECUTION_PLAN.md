# 🚀 Updated Analysis Execution Plan (2021-2025)
**Date**: October 13, 2025
**Updated Timeline**: Jan 1, 2021 → Jul 1, 2025 (4.5 years)

---

## 📋 Executive Summary

### What's Changing
- **Old timeline**: 2019-2024
- **New timeline**: Jan 1, 2021 → Jul 1, 2025 (4.5 years)
- **Key updates**:
  - ✅ Download 2025 news data (Jan-Jul 2025)
  - ✅ Apply 3:30 PM ET cutoff rule (news after 3:30 PM counts for next day)
  - ✅ Re-filter and categorize all news
  - ✅ Re-run event study with updated dates
  - ✅ Generate new visualizations and summaries

### Scripts Created
1. ✅ [32_download_2025_news_update.py](02-scripts/32_download_2025_news_update.py) - Download 2025 news
2. ✅ [33_updated_news_filter_with_timing.py](02-scripts/33_updated_news_filter_with_timing.py) - Filter with timing fix
3. ✅ [34_master_updated_analysis_pipeline.py](02-scripts/34_master_updated_analysis_pipeline.py) - Master orchestrator
4. ✅ [35_generate_category_events_2021_2025.py](02-scripts/35_generate_category_events_2021_2025.py) - Category events
5. ⏳ [36_verify_fama_french_2021_2025.py](02-scripts/36_verify_fama_french_2021_2025.py) - **TO CREATE**
6. ⏳ [37_event_study_2021_2025_all_stocks.py](02-scripts/37_event_study_2021_2025_all_stocks.py) - **TO CREATE**
7. ⏳ [38_check_news_day_balance.py](02-scripts/38_check_news_day_balance.py) - **TO CREATE**
8. ⏳ [39_aggregate_summaries_2021_2025.py](02-scripts/39_aggregate_summaries_2021_2025.py) - **TO CREATE**
9. ⏳ [40_sector_visualizations_2021_2025.py](02-scripts/40_sector_visualizations_2021_2025.py) - **TO CREATE**

---

## 🎯 STEP-BY-STEP EXECUTION GUIDE

### Prerequisites

1. **Environment Setup**
```bash
cd /Users/vivan/Desktop/Central\ File\ Manager/02\ USC/04\ Semester\ 3/03\ DSO\ 585\ -\ Data\ Driven\ Consulting/01\ Project/news-stock-prediction

# Activate environment
source dso-585-datadriven/bin/activate
# OR
/Users/vivan/miniconda3/envs/DSO530/bin/python
```

2. **Verify API Keys**
```bash
cat .env | grep EODHD_API_KEY
```

---

## 📦 STEP 1: Download 2025 News Data (30-60 minutes)

### What it does:
- Downloads news from Jan 1, 2025 to Jul 1, 2025
- Merges with existing data (2019-2024)
- Applies 3:30 PM ET cutoff rule
- Creates backups before updating

### Execute:
```bash
cd 02-scripts
/Users/vivan/miniconda3/envs/DSO530/bin/python 32_download_2025_news_update.py
```

### Expected Output:
```
================================================================================
DOWNLOADING 2025 NEWS DATA UPDATE
================================================================================
Date range: 2025-01-01 to 2025-07-01
Stocks: 50

[Creating backup...]
✅ Backed up 50 files to ../backup_old_data

[Processing each stock...]
[1/50] AAPL - Apple (Technology)
  ✓ Loaded existing data: 25,275 articles
    Date range: 2020-03-23 to 2024-01-31
  → Downloading 2025 news data...
  ✓ Downloaded 2025 data: 5,432 articles
  ✓ Merged data: 30,707 total articles
    Final date range: 2020-03-23 to 2025-06-30
  → Applying 3:30 PM ET cutoff rule...
    • Shifted 12,453 articles (40.5%) to next day
  ✅ Saved updated file

[...]

UPDATE SUMMARY
================================================================================
✅ Successfully updated: 49/50
⚠️  No new data: 1 (TSLA already has 2025 data)
❌ Failed: 0
```

### Verification:
```bash
cd ../01-data
/Users/vivan/miniconda3/envs/DSO530/bin/python << 'EOF'
import pandas as pd
import glob

for file in sorted(glob.glob('*_eodhd_news.csv'))[:5]:
    ticker = file.replace('_eodhd_news.csv', '')
    df = pd.read_csv(file)
    df['date'] = pd.to_datetime(df['date'])
    df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')

    print(f"{ticker}: {len(df)} articles, {df['date'].min()} to {df['date'].max()}")
    if 'event_date' in df.columns:
        shifted = (df['event_date'] != df['date']).sum()
        print(f"  Event dates shifted: {shifted} ({shifted/len(df)*100:.1f}%)")
EOF
```

---

## 🔍 STEP 2: Filter and Categorize News (15-20 minutes)

### What it does:
- Filters news for 2021-2025 window
- Applies balanced filter (sentiment + category + quality)
- Categorizes into 8 event types
- One event per day (highest priority + strongest sentiment)

### Execute:
```bash
cd /Users/vivan/Desktop/Central\ File\ Manager/02\ USC/04\ Semester\ 3/03\ DSO\ 585\ -\ Data\ Driven\ Consulting/01\ Project/news-stock-prediction/02-scripts
/Users/vivan/miniconda3/envs/DSO530/bin/python 33_updated_news_filter_with_timing.py
```

### Expected Output:
```
================================================================================
UPDATED NEWS FILTERING WITH 3:30 PM ET CUTOFF
================================================================================
Analysis window: 2021-01-01 to 2025-07-01
Stocks: 50

[1/50] AAPL - Apple (Technology)
============================================================
  ✓ Loaded: 30,707 articles
    Date range: 2020-03-23 to 2025-06-30
    Analysis window (2021-2025): 28,543 articles
  → Categorizing events...
    Category distribution:
      Market Performance: 14,523 (50.9%)
      Earnings: 3,421 (12.0%)
      Analyst Ratings: 2,145 (7.5%)
      Product Launch: 1,834 (6.4%)
      ...
  → Detecting false positives...
    • Ticker in title: 4,523 (15.8%)
    • Single ticker: 8,234 (28.8%)
    • FP Score 0 (clean): 2,145 (7.5%)
  → Applying balanced filter...
    • Before filter: 28,543 articles
    • After filter: 1,234 articles (4.3%)
    • One per day: 876 events
  ✅ Saved: ../03-output/news_filtering_2021_2025/AAPL_balanced_filtered.csv

[...]

FILTERING SUMMARY
================================================================================
✅ Successfully processed: 50/50 stocks
Total articles: 1,427,150
Total filtered events: 43,789
Average events per stock: 875.8
Median events per stock: 823
```

### Verification:
```bash
cd ../03-output/news_filtering_2021_2025
ls -1 *.csv | wc -l  # Should be 51 (50 stocks + 1 summary)
head -5 filtering_summary.csv
```

---

## 📅 STEP 3: Generate Category Event Dates (5-10 minutes)

### What it does:
- Extracts event dates for each category
- Creates 8 event files per stock (one per category)
- Total: 400 event files (50 stocks × 8 categories)

### Execute:
```bash
cd /Users/vivan/Desktop/Central\ File\ Manager/02\ USC/04\ Semester\ 3/03\ DSO\ 585\ -\ Data\ Driven\ Consulting/01\ Project/news-stock-prediction/02-scripts
/Users/vivan/miniconda3/envs/DSO530/bin/python 35_generate_category_events_2021_2025.py
```

### Expected Output:
```
================================================================================
GENERATE CATEGORY EVENT DATES (2021-2025)
================================================================================
Stocks: 50
Categories: 8

[1/50] AAPL - Apple (Technology)
============================================================
  ✓ Loaded: 876 filtered events
  ✓ Earnings: 102 events → Earnings_events.csv
  ✓ Product Launch: 87 events → Product_Launch_events.csv
  ✓ Analyst Ratings: 143 events → Analyst_Ratings_events.csv
  ...

GENERATION SUMMARY
================================================================================
✅ Successfully processed: 50/50 stocks

Category statistics across all stocks:
Category                  Total Events    Avg/Stock       Stocks with Events
--------------------------------------------------------------------------------
Earnings                  5,123           102.5           50/50
Analyst Ratings           7,156           143.1           50/50
Product Launch            4,345           86.9            50/50
...
```

### Verification:
```bash
cd ../03-output/event_study_2021_2025/event_dates
ls -d */ | wc -l  # Should be 50
ls AAPL/*.csv     # Should show 8-9 files
```

---

## 📊 STEP 4: Verify Fama-French Data (1-2 minutes)

### What it does:
- Checks if Fama-French data covers 2021-2025
- Downloads if missing
- Verifies factor data quality

### Execute:
```bash
cd /Users/vivan/Desktop/Central\ File\ Manager/02\ USC/04\ Semester\ 3/03\ DSO\ 585\ -\ Data\ Driven\ Consulting/01\ Project/news-stock-prediction/02-scripts

# First check what we have
/Users/vivan/miniconda3/envs/DSO530/bin/python << 'EOF'
import pandas as pd
from pathlib import Path

ff_file = Path('../01-data/fama_french_factors.csv')
if ff_file.exists():
    df = pd.read_csv(ff_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Fama-French data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total days: {len(df)}")

    # Check 2021-2025 coverage
    df_2021_2025 = df[(df['date'] >= '2021-01-01') & (df['date'] <= '2025-07-01')]
    print(f"\n2021-2025 coverage: {len(df_2021_2025)} days")

    if len(df_2021_2025) > 1000:
        print("✅ Sufficient coverage for analysis")
    else:
        print("⚠️  May need to download updated data")
else:
    print("❌ Fama-French data file not found")
EOF
```

### Expected Output:
```
Fama-French data range: 2019-01-01 to 2025-06-30
Total days: 2372

2021-2025 coverage: 1640 days
✅ Sufficient coverage for analysis
```

### If data needs updating:
```bash
/Users/vivan/miniconda3/envs/DSO530/bin/python << 'EOF'
import pandas_datareader as pdr
from datetime import datetime

ff_factors = pdr.DataReader(
    'F-F_Research_Data_5_Factors_2x3_daily',
    'famafrench',
    start='2021-01-01',
    end='2025-07-01'
)[0] / 100

ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m%d')
ff_factors.to_csv('../01-data/fama_french_factors_2021_2025.csv')
print(f"✅ Downloaded {len(ff_factors)} days of Fama-French data")
EOF
```

---

## 🔬 STEP 5: Run Event Study Analysis (60-90 minutes)

### What it does:
- Runs Fama-French 5-factor event study for all 50 stocks
- Uses 2021-2025 analysis window
- Generates abnormal returns for news vs non-news days
- Creates individual stock visualizations

### Status: ⚠️ **SCRIPT NEEDS TO BE CREATED**

This should be adapted from [26_robust_event_study_50_stocks.py](02-scripts/26_robust_event_study_50_stocks.py) with:
- Updated date range (2021-2025)
- Updated input path (news_filtering_2021_2025/)
- Updated output path (event_study_2021_2025/)
- Use event_date column instead of date

---

## ⚖️ STEP 6: Check News Day Balance (2-3 minutes)

### What it does:
- Counts # news days vs # non-news days for each stock
- Verifies balanced filter creates reasonable distribution
- Target: 20-30% news days, 70-80% non-news days

### Status: ⚠️ **SCRIPT NEEDS TO BE CREATED**

### What to check:
```python
# For each stock:
total_trading_days = count_trading_days('2021-01-01', '2025-07-01')
news_days = len(filtered_events)
non_news_days = total_trading_days - news_days

news_percentage = news_days / total_trading_days * 100

# Want: 20-30% news days
if 15 <= news_percentage <= 35:
    print(f"✅ {ticker}: Balanced ({news_percentage:.1f}% news days)")
else:
    print(f"⚠️  {ticker}: Imbalanced ({news_percentage:.1f}% news days)")
```

---

## 📈 STEP 7: Generate Aggregate Summaries (5-10 minutes)

### What it does:
- Combines results across all 50 stocks
- Creates 8 aggregate visualizations
- Generates comprehensive summary CSV

### Status: ⚠️ **SCRIPT NEEDS TO BE CREATED**

Adapt from [30_aggregate_results_and_visualizations.py](02-scripts/30_aggregate_results_and_visualizations.py)

---

## 🏢 STEP 8: Generate Sector Visualizations (5-10 minutes)

### What it does:
- Creates sector-level analysis (9 sectors)
- 5 charts per sector (45 total)
- Sector comparison heatmaps

### Status: ⚠️ **SCRIPT NEEDS TO BE CREATED**

Adapt from [31_sector_analysis_visualizations.py](02-scripts/31_sector_analysis_visualizations.py)

---

## 🚀 AUTOMATED EXECUTION (Recommended)

### Run Master Pipeline:
```bash
cd /Users/vivan/Desktop/Central\ File\ Manager/02\ USC/04\ Semester\ 3/03\ DSO\ 585\ -\ Data\ Driven\ Consulting/01\ Project/news-stock-prediction/02-scripts

/Users/vivan/miniconda3/envs/DSO530/bin/python 34_master_updated_analysis_pipeline.py
```

This will:
1. Show each step before executing
2. Ask for confirmation
3. Track progress
4. Provide summary at the end

---

## 📊 EXPECTED FINAL OUTPUT

### Directory Structure:
```
03-output/
├── news_filtering_2021_2025/
│   ├── AAPL_balanced_filtered.csv (50 stocks)
│   ├── filtering_summary.csv
│   └── ...
│
├── event_study_2021_2025/
│   ├── event_dates/
│   │   ├── AAPL/
│   │   │   ├── Earnings_events.csv
│   │   │   ├── Analyst_Ratings_events.csv
│   │   │   └── ... (8 categories)
│   │   └── ... (50 stocks)
│   │
│   ├── results/
│   │   ├── AAPL/
│   │   │   ├── summary.csv
│   │   │   ├── abnormal_returns.csv
│   │   │   └── event_study_visualization.png
│   │   └── ... (50 stocks)
│   │
│   ├── aggregate_summary/
│   │   ├── 01_ar_comparison_all_stocks.png
│   │   ├── 02_effect_size_all_stocks.png
│   │   ├── ... (8 charts)
│   │   └── full_summary.csv
│   │
│   └── sector_analysis/
│       ├── Technology/
│       │   ├── 01_ar_comparison.png
│       │   ├── ... (5 charts)
│       │   └── sector_summary.csv
│       └── ... (9 sectors)
```

### Key Metrics to Report:
1. **Total articles analyzed**: ~1.4M
2. **Filtered events**: ~44,000
3. **% stocks with significant effects**: Target 30-40%
4. **Average effect size (Cohen's d)**: Target 0.05-0.15
5. **News day balance**: Target 20-30% news days

---

## ⚠️ CRITICAL NOTES

### 3:30 PM ET Cutoff Rule
```
Example:
- News published at 2:00 PM ET on Jan 5 → Event date: Jan 5
- News published at 4:30 PM ET on Jan 5 → Event date: Jan 6

Rationale:
- Market closes at 4:00 PM ET
- News after 3:30 PM cannot be fully priced in same day
- Investors react next trading day
```

### News Day Balance
- **Too many news days (>40%)**: Filter too lenient, may include noise
- **Too few news days (<15%)**: Filter too strict, may miss real events
- **Target**: 20-30% news days

### Date Range Choice
- **2021-2025** chosen because:
  - EODHD news API has best coverage starting 2021
  - Includes COVID recovery period
  - Includes 2022-2023 rate hike cycle
  - Includes recent 2024-2025 AI boom
  - 4.5 years = sufficient sample size

---

## 🐛 TROUBLESHOOTING

### Issue: "No module named pandas"
```bash
/Users/vivan/miniconda3/envs/DSO530/bin/pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### Issue: "EODHD API key not found"
```bash
cat .env | grep EODHD
# If missing, add: EODHD_API_KEY=your_key_here
```

### Issue: "Rate limit exceeded"
- EODHD free tier: 20 requests/day
- EODHD paid tier: 100,000 requests/day
- Wait 24 hours or upgrade

### Issue: "Fama-French download fails"
```bash
/Users/vivan/miniconda3/envs/DSO530/bin/pip install pandas-datareader --upgrade
```

---

## ✅ COMPLETION CHECKLIST

- [ ] Step 1: Downloaded 2025 news for all 50 stocks
- [ ] Step 2: Filtered and categorized news (2021-2025)
- [ ] Step 3: Generated 400 category event files
- [ ] Step 4: Verified Fama-French data coverage
- [ ] Step 5: Ran event study for all 50 stocks
- [ ] Step 6: Checked news day balance (all stocks 15-35%)
- [ ] Step 7: Generated 8 aggregate visualizations
- [ ] Step 8: Generated 45 sector visualizations
- [ ] Verified all output files exist
- [ ] Reviewed summary statistics
- [ ] Created executive summary document
- [ ] Committed changes to Git

---

## 📞 NEXT STEPS AFTER COMPLETION

1. **Review Results**
   - Check significance rates
   - Identify top-performing categories
   - Note sector patterns

2. **Update Documentation**
   - Update README.md with new timeline
   - Update presentation slides
   - Add 2025 data note to all reports

3. **Create Final Deliverables**
   - Executive summary (1-page)
   - Presentation deck (15-20 slides)
   - Academic paper draft (optional)

---

**Last Updated**: October 13, 2025, 6:35 PM PST
