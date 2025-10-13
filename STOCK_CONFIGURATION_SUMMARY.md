# Stock Configuration Summary

## ✅ Configuration Status

**Total Stocks**: 54
**No Duplicates**: ✓ Validated
**Minimum per Sector**: 5 stocks
**Sectors**: 10

---

## 📊 Complete Stock List by Sector

### 1. Technology (XLK) - 7 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| AAPL | Apple | Consumer Electronics |
| MSFT | Microsoft | Software |
| NVDA | NVIDIA | Semiconductors |
| AVGO | Broadcom | Semiconductors |
| ORCL | Oracle | Enterprise Software |
| CRM | Salesforce | Cloud/SaaS |
| ADBE | Adobe | Creative Software |

**Data Status**: AAPL ✓, MSFT ✓, NVDA ✓ | AVGO ✗, ORCL ✗, CRM ✗, ADBE ✗

---

### 2. Finance (XLF) - 6 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| JPM | JPMorgan Chase | Banking |
| BAC | Bank of America | Banking |
| WFC | Wells Fargo | Banking |
| GS | Goldman Sachs | Investment Banking |
| MS | Morgan Stanley | Investment Banking |
| BLK | BlackRock | Asset Management |

**Data Status**: JPM ✓, GS ✓ | BAC ✗, WFC ✗, MS ✗, BLK ✗

---

### 3. Healthcare (XLV) - 6 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| JNJ | Johnson & Johnson | Pharmaceuticals |
| UNH | UnitedHealth | Health Insurance |
| PFE | Pfizer | Pharmaceuticals |
| ABBV | AbbVie | Biopharmaceuticals |
| TMO | Thermo Fisher | Life Sciences |
| LLY | Eli Lilly | Pharmaceuticals |

**Data Status**: JNJ ✓, PFE ✓ | UNH ✗, ABBV ✗, TMO ✗, LLY ✗

---

### 4. Communication Services (XLC) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| GOOGL | Alphabet | Internet Services |
| META | Meta | Social Media |
| NFLX | Netflix | Streaming |
| DIS | Disney | Entertainment |
| CMCSA | Comcast | Telecom/Media |

**Data Status**: GOOGL ✓, META ✓ | NFLX ✗, DIS ✗, CMCSA ✗

---

### 5. Consumer Discretionary (XLY) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| AMZN | Amazon | E-commerce/Cloud |
| TSLA | Tesla | Electric Vehicles |
| HD | Home Depot | Home Improvement |
| MCD | McDonald's | Fast Food |
| NKE | Nike | Apparel |

**Data Status**: AMZN ✓, TSLA ✓ | HD ✗, MCD ✗, NKE ✗

---

### 6. Consumer Staples (XLP) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| PG | Procter & Gamble | Consumer Products |
| KO | Coca-Cola | Beverages |
| PEP | PepsiCo | Food & Beverages |
| WMT | Walmart | Retail |
| COST | Costco | Wholesale Retail |

**Data Status**: PG ✓, WMT ✓ | KO ✗, PEP ✗, COST ✗

---

### 7. Industrials (XLI) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| BA | Boeing | Aerospace |
| CAT | Caterpillar | Heavy Machinery |
| UNP | Union Pacific | Rail Transport |
| HON | Honeywell | Conglomerate |
| GE | General Electric | Industrial |

**Data Status**: BA ✓ | CAT ✗, UNP ✗, HON ✗, GE ✗

---

### 8. Energy (XLE) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| XOM | Exxon Mobil | Oil & Gas |
| CVX | Chevron | Oil & Gas |
| COP | ConocoPhillips | Oil & Gas |
| SLB | Schlumberger | Oil Services |
| EOG | EOG Resources | Oil & Gas Exploration |

**Data Status**: XOM ✓ | CVX ✗, COP ✗, SLB ✗, EOG ✗

---

### 9. Utilities (XLU) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| NEE | NextEra Energy | Utilities |
| DUK | Duke Energy | Utilities |
| SO | Southern Company | Utilities |
| D | Dominion Energy | Utilities |
| AEP | American Electric Power | Utilities |

**Data Status**: NEE ✓ | DUK ✗, SO ✗, D ✗, AEP ✗

---

### 10. Real Estate (XLRE) - 5 stocks
| Ticker | Company | Sub-sector |
|--------|---------|------------|
| AMT | American Tower | Cell Towers |
| PLD | Prologis | Industrial REITs |
| CCI | Crown Castle | Infrastructure |
| EQIX | Equinix | Data Center REITs |
| SPG | Simon Property | Retail REITs |

**Data Status**: AMT ✓ | PLD ✗, CCI ✗, EQIX ✗, SPG ✗

---

## 📈 Data Collection Progress

### Stock Price Data
- **Have**: 17/54 (31%)
- **Need**: 37/54 (69%)

### News Data
- **Have**: 17/54 (31%)
- **Need**: 37/54 (69%)

### ETF Data
- **Have**: 10/10 (100%) ✓

---

## 🎯 Stocks to Download (37 total)

```python
MISSING_STOCKS = [
    # Technology (4)
    'AVGO', 'ORCL', 'CRM', 'ADBE',

    # Finance (4)
    'BAC', 'WFC', 'MS', 'BLK',

    # Healthcare (4)
    'UNH', 'ABBV', 'TMO', 'LLY',

    # Communication (3)
    'NFLX', 'DIS', 'CMCSA',

    # Consumer Discretionary (3)
    'HD', 'MCD', 'NKE',

    # Consumer Staples (3)
    'KO', 'PEP', 'COST',

    # Industrials (4)
    'CAT', 'UNP', 'HON', 'GE',

    # Energy (4)
    'CVX', 'COP', 'SLB', 'EOG',

    # Utilities (4)
    'DUK', 'SO', 'D', 'AEP',

    # Real Estate (4)
    'PLD', 'CCI', 'EQIX', 'SPG'
]
```

---

## 🔍 Why This Configuration?

### Diversity
- **10 sectors** covering entire market
- **54 stocks** for statistical power
- **5-7 stocks per sector** for within-sector analysis

### Balance
- Each sector adequately represented
- Mix of large-cap, stable companies
- Different business models within sectors

### Research Benefits
1. **Within-sector comparisons**: Does news affect tech stocks differently?
2. **Cross-sector insights**: Which sectors are most news-sensitive?
3. **Robustness**: Results not dependent on single stock
4. **Portfolio construction**: Can build sector-balanced portfolios

---

## 📋 Next Immediate Steps

1. **Download missing data** (Priority 1)
   - Run data collection scripts for 37 stocks
   - Verify data quality

2. **Validate completeness** (Priority 2)
   - Check all 54 stocks have 2022-2024 data
   - Ensure no major gaps

3. **Begin experimentation** (Priority 3)
   - Feature engineering across all stocks
   - Baseline model development
   - Sector-level analysis

---

**Configuration File**: `02-scripts/16_expanded_stock_config.py`
**Full Experiment Plan**: `EXPERIMENT_PLAN.md`
**Last Updated**: 2025-10-12
