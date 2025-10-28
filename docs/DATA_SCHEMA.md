# Data Schema Documentation

This document describes the required data format for the TooSharpe Institutional Dashboard.

## Input Files

All data files should be placed in the `data/` directory as CSV files.

### 1. PositionLevelPNLAndExposure.csv

Position-level P&L and exposure data. Must contain daily position-level information.

**Required Columns:**
- `Date`: Date of the position (YYYY-MM-DD format)
- `Account`: Account identifier (string)
- `SEDOL`: SEDOL identifier for the security
- `PNL`: Daily P&L for the position
- `GMV`: Gross Market Value (absolute dollar exposure)
- `NMV`: Net Market Value (signed dollar exposure)

**Optional Columns:**
- Any other metadata columns (will be preserved through the pipeline)

**Sample Data:**
```
Date,Account,SEDOL,PNL,GMV,NMV
2022-01-01,Account_A,B1234567,1000,1000000,800000
2022-01-01,Account_A,B2345678,-500,800000,-400000
```

### 2. ReferenceData.csv

Security reference data including sector classifications, market cap, and liquidity metrics.

**Required Columns:**
- `Date`: Date of the reference data (YYYY-MM-DD)
- `SEDOL`: SEDOL identifier (must match positions)
- `GICS_SECTOR`: GICS sector classification
- `MARKET_CAP`: Market capitalization in dollars
- `AverageDailyVolume_20_DAY`: 20-day average daily trading volume
- `ETF_C`: ETF flag (1 for ETF, 0 otherwise)

**Optional Columns:**
- `GICS_SECTOR_CLEAN`: Cleaned sector name (auto-generated if missing)
- `Side`: Long/Short indicator
- Other classification columns

**Sample Data:**
```
Date,SEDOL,GICS_SECTOR,MARKET_CAP,AverageDailyVolume_20_DAY,ETF_C
2022-01-01,B1234567,Information Technology,50000000000,10000000,0
2022-01-01,B2345678,Financials,20000000000,5000000,0
```

### 3. AccountInformation.csv

Account metadata including static notional amounts.

**Required Columns:**
- `Account`: Account identifier
- `StaticNotional`: Static notional amount for the account (used for leverage calculations)

**Optional Columns:**
- Any additional account-level metadata

**Sample Data:**
```
Account,StaticNotional
Account_A,50000000
Account_B,30000000
```

### 4. AUM.csv

Assets Under Management data.

**Required Columns:**
- `AsOfDate` or `Date`: Date (YYYY-MM-DD)
- `AUM`: Assets under management value

**Sample Data:**
```
AsOfDate,AUM
2022-01-01,100000000
2022-02-01,105000000
```

### 5. DailyIndexReturns.csv

Benchmark index returns data.

**Required Columns:**
- `Date`: Date (YYYY-MM-DD)
- One or more benchmark columns (e.g., `ACWI`, `SP500`, etc.)

**Sample Data:**
```
Date,ACWI,SP500
2022-01-01,0.001,0.0008
2022-01-02,-0.0005,-0.0003
```

## Data Quality Requirements

### 1. Date Format
- All dates should be in ISO format: YYYY-MM-DD
- The pipeline will attempt to parse other formats, but YYYY-MM-DD is recommended

### 2. Numeric Columns
- Numeric values should not contain currency symbols or commas
- Use standard decimal notation (e.g., 1000.50, not $1,000.50)
- Missing values should be empty or "NA"

### 3. String Columns
- Trailing/leading whitespace is automatically trimmed
- Common null tokens (null, na, none, empty string) are converted to NaN

### 4. Data Relationships
- **SEDOL consistency**: SEDOLs in ReferenceData should match those in Position data
- **Account consistency**: Accounts in AccountInformation should match those in Positions
- **Date ranges**: All files should have overlapping date ranges for proper integration

## Data Validation

The pipeline performs the following validations:

1. **Required columns**: Checks for presence of all required columns
2. **Singleton SEDOLs**: Removes SEDOLs that appear only once (likely data errors)
3. **Missing reference data**: Flags positions without matching reference data
4. **Date parsing**: Attempts to parse dates with fallback error handling
5. **Type coercion**: Converts numeric columns with error tolerance

## Output Files

After running the pipeline, the following outputs are generated:

### Cleaned Files (`outputs/cleaned/`)
- `ReferenceData_clean.csv`
- `PositionLevelPNLAndExposure_clean.csv`
- `AccountInformation_clean.csv`
- `AUM_clean.csv`
- `DailyIndexReturns_returns.csv`

### Integrated Files (`outputs/integrated/`)
- `TooSharpe_InternalIntegrated.csv`: Joined positions + reference + accounts

### Intermediary Files (`outputs/intermediary/`)
- `intermediary.csv` and `intermediary.parquet`: Analytics-ready dataset with computed metrics
- Contains derived columns:
  - `FLOW`: Cash flows
  - `RET_TWR`: Time-weighted returns
  - `W_IDX`: Wealth index
  - `LEVERAGE_ACCT`: Account-level leverage
  - `LEVERAGE_FUND`: Fund-level leverage
  - `W_NAV`, `W_GROSS`, `W_NETABS`: Position weights
  - `RC_NAV`, `PNL_CONTRIB_NAV`: Return contributions
  - `ADV_COVER_GMV`, `DAYS_TO_LIQUIDATE`: Liquidity metrics
  - And more...

## Troubleshooting

### Error: "Missing required columns"
- Ensure all required columns exist in your input files
- Check for typos in column names (case-sensitive)

### Error: "No intermediary file found"
- Run the data pipeline first using `python scripts/run_pipeline.py`

### Missing data in dashboard
- Check the `outputs/dropped/` folder for records that were excluded
- Review the data quality logs for warnings

### Performance issues
- Large datasets (>1M rows) may take several minutes to process
- Consider filtering to a specific date range for faster iteration

