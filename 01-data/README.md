# 📊 Data Directory

This directory contains all data files used in the news-stock prediction analysis.

## 📁 Directory Structure

```
01-data/
├── Stock Data
│   ├── AAPL_stock_data.csv
│   └── TSLA_stock_data.csv
│
├── Market Data
│   ├── sp500_market_data.csv
│   ├── fama_french_factors.csv
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv
│   ├── XLK_sector_factor.csv (Technology sector)
│   └── XLY_sector_factor.csv (Consumer Discretionary sector)
│
├── News Data (Raw)
│   ├── AAPL_eodhd_news.csv (738,103 articles)
│   ├── TSLA_eodhd_news.csv (1,407,023 articles)
│   ├── AAPL_finnhub_historical.csv
│   └── TSLA_finnhub_historical.csv
│
└── Event Dates (Filtered)
    ├── AAPL_improved_events.csv (2,357 events)
    ├── AAPL_improved_event_dates.csv (33 final dates)
    ├── TSLA_improved_events.csv (1,923 events)
    ├── TSLA_improved_event_dates.csv (23 final dates)
    ├── AAPL_major_events.csv
    ├── AAPL_major_event_dates.csv
    ├── TSLA_major_events.csv
    └── TSLA_major_event_dates.csv
```

## 📝 File Descriptions

### Stock Price Data

**Format**: CSV with columns `[Date, Open, High, Low, Close, Adj Close, Volume]`

**`AAPL_stock_data.csv`**
- **Ticker**: Apple Inc. (AAPL)
- **Period**: 2020-08-24 to 2024-10-10 (~4.1 years)
- **Trading Days**: 1,025
- **Source**: Yahoo Finance via yfinance
- **Adjustments**: Split-adjusted and dividend-adjusted

**`TSLA_stock_data.csv`**
- **Ticker**: Tesla Inc. (TSLA)
- **Period**: 2019-01-02 to 2024-10-10 (~5.8 years)
- **Trading Days**: 1,422
- **Source**: Yahoo Finance via yfinance
- **Adjustments**: Split-adjusted and dividend-adjusted

### Factor Data

**`fama_french_factors.csv`**
- **Source**: Kenneth French Data Library
- **Factors**: Mkt-RF, SMB, HML, RMW, CMA, RF
- **Frequency**: Daily
- **Units**: Decimal (e.g., 0.01 = 1%)
- **Description**:
  - **Mkt-RF**: Market risk premium (Market return - Risk-free rate)
  - **SMB**: Small Minus Big (Small cap - Large cap)
  - **HML**: High Minus Low (Value - Growth)
  - **RMW**: Robust Minus Weak (High profitability - Low profitability)
  - **CMA**: Conservative Minus Aggressive (Low investment - High investment)
  - **RF**: Risk-free rate (1-month Treasury bill)

**`XLK_sector_factor.csv`** & **`XLY_sector_factor.csv`**
- **Source**: Yahoo Finance (sector ETFs)
- **Purpose**: Sector-specific risk factors
- **XLK**: Technology Select Sector SPDR Fund (for AAPL)
- **XLY**: Consumer Discretionary Select Sector SPDR Fund (for TSLA)

### News Data (Raw)

**`AAPL_eodhd_news.csv`** (738,103 articles)
- **Columns**: `[ticker, date, title, content, link, symbols, tags, sentiment_polarity, sentiment_neg, sentiment_neu, sentiment_pos, content_length]`
- **Source**: EODHD Financial News API
- **Period**: 2019-2024
- **Coverage**: All articles mentioning AAPL
- **Sentiment**: VADER sentiment analysis scores

**`TSLA_eodhd_news.csv`** (1,407,023 articles)
- **Columns**: Same as AAPL
- **Source**: EODHD Financial News API
- **Period**: 2019-2024
- **Coverage**: All articles mentioning TSLA
- **Note**: TSLA has ~2x more articles due to higher media coverage

### Event Dates (Filtered)

**`AAPL_improved_events.csv`** (2,357 events)
- **Filtering**: Multi-stage pipeline (sentiment + category + volume)
- **Purpose**: Intermediate filtered events before final selection

**`AAPL_improved_event_dates.csv`** (33 final dates)
- **Filtering**: One event per day, aligned with trading days
- **Event Density**: 3.2% (33/1,025 days)
- **Usage**: Primary event dates for final analysis
- **Columns**: `[Date]` or `[Date, Title, Sentiment, Category, ...]`

**TSLA Versions**: Same structure, 23 final events (1.6% density)

## 🔄 Data Collection Process

### 1. Stock Data Collection
```python
import yfinance as yf

# Download stock data
ticker = yf.Ticker("AAPL")
data = ticker.history(start="2020-01-01", end="2024-10-10")
data.to_csv("AAPL_stock_data.csv")
```

### 2. Factor Data Collection
```python
import pandas_datareader as pdr

# Download Fama-French factors
ff_factors = pdr.get_data_famafrench(
    'F-F_Research_Data_5_Factors_2x3_daily',
    start='2019', end='2024'
)[0]
ff_factors.to_csv("fama_french_factors.csv")
```

### 3. News Data Collection
```python
import requests

# EODHD API
url = f"https://eodhistoricaldata.com/api/news"
params = {
    'api_token': API_KEY,
    's': 'AAPL.US',
    'from': '2019-01-01',
    'to': '2024-10-10',
    'limit': 1000
}
response = requests.get(url, params=params)
news_data = response.json()
```

## 📊 Data Statistics

### Stock Data

| Ticker | Start Date | End Date | Trading Days | Mean Daily Return | Volatility (Annualized) |
|--------|-----------|----------|--------------|-------------------|-------------------------|
| AAPL | 2020-08-24 | 2024-10-10 | 1,025 | +0.12% | 28.5% |
| TSLA | 2019-01-02 | 2024-10-10 | 1,422 | +0.15% | 62.3% |

### News Data

| Ticker | Raw Articles | Filtered Events | Final Events | Filtering Ratio |
|--------|-------------|-----------------|--------------|-----------------|
| AAPL | 738,103 | 2,357 | 33 | 0.0045% |
| TSLA | 1,407,023 | 1,923 | 23 | 0.0016% |

### Sentiment Distribution (Sample)

**AAPL** (first 100 filtered events):
- Mean Polarity: -0.03
- Std Dev: 0.65
- Range: [-0.98, +0.95]

**TSLA** (first 100 filtered events):
- Mean Polarity: -0.08
- Std Dev: 0.58
- Range: [-0.92, +0.88]

## ⚠️ Data Usage Notes

### Important Considerations

1. **Data Size**: News CSV files are very large (100MB+ for AAPL, 200MB+ for TSLA)
   - Git LFS recommended for version control
   - Consider providing download links instead of committing to repo

2. **Data Privacy**: Check API terms of service before sharing raw news data
   - EODHD data may have redistribution restrictions
   - Consider sharing only event dates, not full articles

3. **Data Updates**: Stock and news data can be updated
   - Run `00_data_acquisition.py` to refresh
   - Results may vary slightly with updated data

4. **Missing Data**: Some dates may have no trading data (holidays, weekends)
   - Scripts handle this automatically
   - Event dates aligned with trading days only

## 🔐 Data Privacy & Security

### Sensitive Files (.gitignore)

The following files are **NOT** included in the Git repository:
- `.env` - Contains API keys
- `*_news_raw.csv` - Large raw news files
- Any files with API credentials

### API Keys Required

To collect data yourself, you need:
- **EODHD API Key**: [Register here](https://eodhistoricaldata.com/)
  - Free tier: Limited requests
  - Paid tier: Unlimited news history
- **No key needed**: Yahoo Finance, Fama-French factors (public data)

## 📥 Download Pre-Collected Data

If you want to skip data collection, download pre-processed data:

```bash
# Coming soon: Download link for pre-collected data
# wget https://example.com/news-stock-data.zip
# unzip news-stock-data.zip -d 01-data/
```

## 🔄 Data Update Procedure

To update data with latest information:

```bash
cd 02-scripts

# Update stock data
python -c "from data_acquisition import update_stock_data; update_stock_data(['AAPL', 'TSLA'])"

# Update news data (requires API key)
python 00_data_acquisition.py --update-news

# Re-run filtering
python create_filtered_news.py
```

## 📚 Data Sources & References

1. **Yahoo Finance**: Stock prices, sector ETFs
   - Free, no registration required
   - High-quality, split-adjusted data
   - API: yfinance Python library

2. **Kenneth French Data Library**: Fama-French factors
   - Free, academic use
   - Daily updates, high-quality
   - URL: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/

3. **EODHD**: Financial news articles
   - Paid API (free tier available)
   - Comprehensive coverage
   - URL: https://eodhistoricaldata.com/

## ❓ FAQ

**Q: Why are there so many news articles?**
A: Raw data includes all articles mentioning the ticker, including market commentary, unrelated mentions, and duplicates. Filtering reduces this to material events.

**Q: Can I use this data for my own research?**
A: Yes, but check each data source's terms of service. Yahoo Finance and Fama-French are free for research. EODHD has usage restrictions.

**Q: How often should I update the data?**
A: For academic research, snapshot at a specific date is fine. For live trading (not recommended based on results!), update daily.

**Q: What if I can't download EODHD data?**
A: The repository includes pre-processed event dates. You can run analysis without raw news data.

---

**Last Updated**: October 11, 2024
