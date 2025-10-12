# EODHD Setup and Run Instructions

Complete guide for downloading all data using EODHD API with your original date ranges.

---

## 📋 Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Bash Commands to Run](#bash-commands-to-run)
3. [File Locations](#file-locations)
4. [Troubleshooting](#troubleshooting)

---

## 🚀 Setup Instructions

### Step 1: Get EODHD API Key

1. Go to https://eodhd.com/register
2. Register for an account
3. Get your API key from the dashboard

**Pricing:**
- **Free Tier:** 20 API calls/day (= 4 news requests)
- **Paid Tier:** $19.99/month (unlimited access to historical news)

### Step 2: Add API Key to .env File

```bash
# Navigate to project root
cd "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction"

# Edit .env file (or create from template)
nano .env
```

Add this line:
```
EODHD_API_KEY=your_actual_api_key_here
```

Save and exit (Ctrl+X, then Y, then Enter)

### Step 3: Activate Virtual Environment

```bash
source dso-585-datadriven/bin/activate
```

---

## 🎯 Bash Commands to Run

### Option A: Download Everything at Once (Master Script)

```bash
# Navigate to scripts folder
cd 02-scripts

# Run master download script
python download_all_eodhd.py
```

This will download:
- ✅ AAPL stock data (2020-01-01 to 2024-01-31)
- ✅ TSLA stock data (2020-01-01 to 2025-10-08)
- ✅ S&P 500 market data (2020-01-01 to 2025-10-08)
- ✅ AAPL news data (via EODHD)
- ✅ TSLA news data (via EODHD)

---

### Option B: Download News Only (Individual Script)

```bash
# Navigate to scripts folder
cd "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/02-scripts"

# Activate virtual environment
source ../dso-585-datadriven/bin/activate

# Run EODHD news downloader
python 00c_eodhd_news.py
```

This will download:
- ✅ AAPL news (2020-01-01 to 2024-01-31)
- ✅ TSLA news (2020-01-01 to 2025-10-08)

---

### Option C: Download Stock Data Only (Using yfinance)

```bash
# In Python interpreter or script
python -c "
import yfinance as yf
import pandas as pd

# AAPL stock data
aapl = yf.Ticker('AAPL')
aapl_df = aapl.history(start='2020-01-01', end='2024-01-31')
aapl_df.to_csv('../01-data/AAPL_stock_data.csv')
print('AAPL stock data downloaded')

# TSLA stock data
tsla = yf.Ticker('TSLA')
tsla_df = tsla.history(start='2020-01-01', end='2025-10-08')
tsla_df.to_csv('../01-data/TSLA_stock_data.csv')
print('TSLA stock data downloaded')

# S&P 500 market data
sp500 = yf.Ticker('^GSPC')
sp500_df = sp500.history(start='2020-01-01', end='2025-10-08')
sp500_df.to_csv('../01-data/sp500_market_data.csv')
print('S&P 500 market data downloaded')
"
```

---

## 📁 File Locations

### Input Files (.env)
```
/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/.env
```

### Scripts Location
```
/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/02-scripts/

Key scripts:
├── 00c_eodhd_news.py          # EODHD news downloader
├── download_all_eodhd.py       # Master download script (all data)
├── 00b_finnhub_news.py         # Finnhub news (alternative)
└── download_finnhub_historical.py  # Finnhub historical (alternative)
```

### Output Files Location
```
/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/01-data/

After running, you'll find:
├── AAPL_eodhd_news.csv        # AAPL news from EODHD
├── TSLA_eodhd_news.csv        # TSLA news from EODHD
├── AAPL_stock_data.csv        # AAPL stock prices
├── TSLA_stock_data.csv        # TSLA stock prices
└── sp500_market_data.csv      # S&P 500 market index
```

---

## 🔧 Troubleshooting

### Issue: "EODHD_API_KEY not found"
**Solution:** Make sure you've added the API key to `.env` file in the project root.

```bash
# Check if .env file exists
ls -la .env

# If not, create from template
cp .env.template .env
nano .env  # Add your API key
```

### Issue: "Payment required (Status 402)"
**Solution:** EODHD free tier has limited historical access. Options:
1. Upgrade to paid plan ($19.99/month)
2. Use Finnhub instead (free, ~1 year of data)
3. Request smaller date ranges

### Issue: "Module not found: pandas"
**Solution:** Activate virtual environment first:
```bash
source dso-585-datadriven/bin/activate
pip install -r requirements.txt
```

### Issue: "Timeout error"
**Solution:** Narrow the date range in the script:
```python
# Edit 00c_eodhd_news.py
# Change to smaller date ranges, e.g.:
start_date='2024-01-01'  # Instead of 2020-01-01
end_date='2024-06-30'    # Smaller range
```

### Issue: Rate limit reached
**Solution:**
- Free tier: 20 API calls/day (4 news requests)
- Wait 24 hours or upgrade to paid plan
- Use Finnhub as backup (60 calls/minute)

---

## 📊 Data Summary

### Original Date Ranges (Your Data):
- **AAPL:** January 2024 only (21 days)
- **TSLA:** 2020-01-01 to 2025-10-08 (1,450 days)

### New Downloads (EODHD):
- **AAPL:** 2020-01-01 to 2024-01-31 (full range)
- **TSLA:** 2020-01-01 to 2025-10-08 (full range)
- **Market:** 2020-01-01 to 2025-10-08 (for beta estimation)

---

## 🎓 Next Steps

After downloading data:

1. **Verify data quality:**
   ```bash
   cd 02-scripts
   python -c "
   import pandas as pd
   aapl = pd.read_csv('../01-data/AAPL_eodhd_news.csv')
   tsla = pd.read_csv('../01-data/TSLA_eodhd_news.csv')
   print(f'AAPL: {len(aapl)} articles')
   print(f'TSLA: {len(tsla)} articles')
   "
   ```

2. **Proceed with event study analysis** (Phase 1)
3. **Use Market Model** for expected returns (simpler than Fama-French)

---

## 💡 Quick Reference

**One-line download command:**
```bash
cd "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/02-scripts" && source ../dso-585-datadriven/bin/activate && python download_all_eodhd.py
```

**Check what's in your data folder:**
```bash
cd "/Users/vivan/Desktop/Central File Manager/02 USC/04 Semester 3/03 DSO 585 - Data Driven Consulting/01 Project/news-stock-prediction/01-data" && ls -lh *.csv
```

**View first few rows of news data:**
```bash
head -5 AAPL_eodhd_news.csv
head -5 TSLA_eodhd_news.csv
```

---

## 📞 Support

- EODHD Docs: https://eodhd.com/financial-apis/stock-market-financial-news-api
- EODHD Support: support@eodhd.com
- Project Issues: Check logs in terminal output
