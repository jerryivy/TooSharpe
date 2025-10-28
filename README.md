# TooSharpe Institutional Dashboard

An institutional-grade performance analytics dashboard for the TooSharpe fund portfolio.

## Features

- **Performance Analysis**: Track portfolio returns, volatility, Sharpe ratio, and benchmark comparison
- **Liquidity Analysis**: Monitor liquidity metrics and position-level liquidity
- **Risk & Attribution**: Analyze sector exposures, market cap distribution, and return attribution
- **PDF Reports**: Generate comprehensive institutional reports

## Getting Started

### Prerequisites

- Python 3.8+
- All Python packages listed in `requirements.txt`

### Installation

1. Clone this repository
2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data Requirements

The application requires the following data files in the `data/` directory:

1. **AccountInformation.csv** - Account metadata
2. **AUM.csv** - Assets under management data
3. **DailyIndexReturns.csv** - Benchmark index returns
4. **PositionLevelPNLAndExposure.csv** - Position-level P&L and exposure data
5. **ReferenceData.csv** - Security reference data (SEDOL, sector, market cap, etc.)

**Required columns in PositionLevelPNLAndExposure.csv:**
- Date, Account, SEDOL, PNL, GMV, NMV

**Required columns in ReferenceData.csv:**
- Date, SEDOL, GICS_SECTOR, MARKET_CAP, AverageDailyVolume_20_DAY, ETF_C

### Running the Data Pipeline

Before running the Streamlit app, you need to process the raw data:

**Option 1: Using the Jupyter Notebook (Interactive)**
```bash
jupyter notebook notebooks/Execution.ipynb
```
Then run all cells to generate the intermediary dataset.

**Option 2: Using the Python Script (Automated)**
```bash
python scripts/run_pipeline.py
```

This will:
1. Clean all source data files
2. Integrate internal tables (Positions + Reference + Accounts)
3. Build the intermediary dataset for analytics
4. Save processed data to `outputs/`

### Running the Application

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

**Data Source (Local Only):**
- The app loads intermediary from `outputs/intermediary/latest/intermediary.parquet` (preferred)
- If Parquet is missing, it falls back to `outputs/intermediary/latest/intermediary.csv`

Ensure one of the above files exists before starting the app.

## Project Structure

```
toosharpe_case_study/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard application
├── data/                          # Source data files (CSV)
│   ├── AccountInformation.csv
│   ├── AUM.csv
│   ├── DailyIndexReturns.csv
│   ├── PositionLevelPNLAndExposure.csv
│   └── ReferenceData.csv
├── notebooks/
│   └── Execution.ipynb           # Data pipeline execution notebook
├── outputs/                       # Generated data
│   ├── cleaned/                   # Cleaned source files
│   ├── integrated/                # Integrated internal data
│   ├── intermediary/             # Analytics-ready intermediary files
│   └── manifest.json             # Processing metadata
├── scripts/
│   └── run_pipeline.py           # Standalone pipeline script
└── src/fund_pipeline/           # Core processing modules
    ├── analytics.py              # Performance, liquidity, risk analytics
    ├── data_handling.py         # Data cleaning and integration
    ├── intermediary_builder.py # Intermediary dataset construction
    └── utils.py                 # Utility functions
```

## Usage Guide

### 1. Data Pipeline Execution

The pipeline processes raw CSV files through several stages:

1. **Cleaning**: Normalizes dates, cleans categorical data, handles missing values
2. **Integration**: Joins positions with reference data and account information
3. **Intermediary Build**: Computes metrics (returns, weights, leverage, liquidity)
4. **Output**: Saves timestamped versions and a `latest/` symlink

### 2. Streamlit Dashboard

The dashboard provides three analysis levels:

- **Fund Level**: Aggregate analysis across all accounts
- **Account Level**: Detailed analysis for one or more selected accounts

Four main tabs:
- **Performance**: Returns, Sharpe ratio, tracking error, benchmark comparison
- **Liquidity**: Liquidity metrics, position liquidity profile
- **Risk & Attribution**: Sector exposure, market cap distribution, P&L by sector
- **Reports**: PDF report generation and data downloads

### 3. Google Drive Integration

**Quick Setup:**
```bash
python setup_gdrive.py
```

**Manual Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Drive API and create OAuth2 credentials
3. Download credentials as `credentials.json` to project root
4. Ensure `intermediary.csv` exists in the Google Drive folder

**Using Google Drive:**
1. Start the Streamlit app
2. In the sidebar, select "Google Drive" as data source
3. Upload `credentials.json` if not already configured
4. The app will automatically load data from Google Drive

See `GDRIVE_SETUP.md` for detailed instructions.

### 4. Generating PDF Reports

1. Navigate to the "Reports & Downloads" tab
2. Configure your analysis (date range, accounts, benchmark)
3. Click "Generate PDF Report"
4. Download the generated institutional-quality PDF report

## Development

### Module Overview

- **`analytics.py`**: Pure analytics functions (no Streamlit dependencies)
  - Performance metrics (Sharpe, tracking error, beta, alpha)
  - Liquidity analysis
  - Risk attribution
  - Benchmark comparison

- **`data_handling.py`**: Data cleaning and integration
  - Cleans each source file independently
  - Integrates internal tables
  - Handles data quality issues (singletons, missing reference data)

- **`intermediary_builder.py`**: Builds analytics-ready intermediary dataset
  - Computes linking metrics (FLOW, RET_TWR, W_IDX)
  - Calculates leverage ratios
  - Computes return contributions
  - Estimates liquidity metrics

## Deployment

This app can be deployed to Streamlit Community Cloud:
1. Push this repository to GitHub
2. Connect to Streamlit Community Cloud
3. Select this repository to deploy

Note: Make sure your data files are included or configured to be loaded from a data source.

## License

Proprietary - TooSharpe Fund
