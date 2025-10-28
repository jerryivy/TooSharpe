# analytics.py
"""
Analytics functions for TooSharpe institutional dashboard.
Pure functions with no Streamlit dependencies.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# IO Functions
# ============================================================================

def load_positions(path: str = "data/PositionLevelPNLAndExposure.csv") -> pd.DataFrame:
    """Load position-level PnL/exposure. Returns sorted DataFrame with Date/Account."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Account"], ignore_index=True)
    return df

def load_reference(path: str = "data/ReferenceData.csv") -> pd.DataFrame:
    """Load reference data."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_accounts(path: str = "data/AccountInformation.csv") -> pd.DataFrame:
    """Load account information."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_aum(path: str = "data/AUM.csv") -> pd.DataFrame:
    """Load AUM data."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Map AsOfDate to Date if needed
    if "AsOfDate" in df.columns and "Date" not in df.columns:
        df["Date"] = pd.to_datetime(df["AsOfDate"], errors="coerce")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

def load_bench(path: str = "data/DailyIndexReturns.csv") -> pd.DataFrame:
    """Load benchmark returns data."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

# ============================================================================
# Helper Functions
# ============================================================================

def _daily_ret_from_positions(positions: pd.DataFrame) -> pd.Series:
    """Compute daily TWR from positions."""
    df = positions.copy()
    
    # Check for RET_TWR first (intermediary file format)
    if "RET_TWR" in df.columns:
        return df.groupby("Date")["RET_TWR"].mean().sort_index()
    
    # Check for PnL/NMV combination
    if "PnL" in df.columns and "NMV" in df.columns:
        df = df.sort_values("Date")
        df["Prev_NMV"] = df.groupby("Account")["NMV"].shift(1)
        grp = df.groupby("Date", as_index=True).agg({"PnL":"sum", "Prev_NMV":"sum"}).sort_index()
        grp["ret"] = grp["PnL"] / grp["Prev_NMV"].replace(0, np.nan)
        return grp["ret"].fillna(0.0)
    
    # Fallback to any return column
    for cand in ["Return", "ret", "PNL"]:
        if cand in df.columns:
            if cand == "PNL":
                # Calculate return from PNL alone
                grp = df.groupby("Date")[cand].sum().sort_index()
                return grp.diff() / grp.shift(1).fillna(1.0)
            return df.groupby("Date")[cand].mean().sort_index()
    
    raise ValueError(f"No suitable return column found. Available columns: {list(df.columns)}")

def _geom_cum(ret: pd.Series) -> pd.Series:
    """Geometric cumulative: cumprod(1+r) - 1."""
    return (1.0 + ret).cumprod() - 1.0

def _rolling_last_12m_mask(dates: pd.DatetimeIndex) -> pd.Index:
    """Return boolean mask for last 252 obs."""
    n = len(dates)
    window = min(252, n)
    mask = pd.Series(False, index=dates)
    if window > 0:
        mask.iloc[-window:] = True
    return mask

def _ann_return_12m(ret: pd.Series) -> float:
    """Annualized geometric return from last ~12M daily series."""
    if ret.empty:
        return float("nan")
    mask = _rolling_last_12m_mask(ret.index)
    r = ret[mask]
    if r.empty:
        return float("nan")
    cum = (1.0 + r).prod() - 1.0
    return (1.0 + cum) ** (252.0 / len(r)) - 1.0

def _ann_vol_12m(ret: pd.Series) -> float:
    """Annualized vol from last ~12M daily returns."""
    if ret.empty:
        return float("nan")
    mask = _rolling_last_12m_mask(ret.index)
    r = ret[mask]
    if r.empty:
        return float("nan")
    return float(np.nanstd(r, ddof=1)) * np.sqrt(252.0)

def _sharpe_12m(ret: pd.Series, rf_annual: float = 0.038) -> float:
    """Sharpe over last ~12M using annualized return and vol."""
    ar = _ann_return_12m(ret)
    av = _ann_vol_12m(ret)
    if np.isnan(ar) or np.isnan(av) or av == 0:
        return float("nan")
    return (ar - rf_annual) / av

def _max_drawdown_from_cum(cum: pd.Series) -> float:
    """Max drawdown from cumulative return series.
    
    Calculate the maximum peak-to-trough decline in value.
    Returns a negative value (e.g., -0.15 means a 15% drawdown).
    """
    if cum.empty or len(cum) < 2:
        return 0.0
    
    try:
        # Create equity curve (1 + cumulative return)
        curve = 1.0 + cum.values
        
        # Calculate running maximum (peak)
        roll_max = np.maximum.accumulate(curve)
        
        # Calculate drawdown from peak: (current - peak) / peak
        dd = (curve - roll_max) / roll_max
        
        # Max drawdown is the most negative value
        # If all values are >= 0 (no drawdown), return 0.0
        max_dd = float(np.nanmin(dd))
        
        # Ensure we return 0 if there was no drawdown (all dd >= 0)
        return max_dd if max_dd <= 0 else 0.0
    except Exception:
        return 0.0

# ============================================================================
# Performance Functions
# ============================================================================

def compute_overview_kpis(
    positions: pd.DataFrame,
    aum: pd.DataFrame,
    bench: pd.DataFrame,
    benchmark_col: Optional[str] = None,
    rf_annual: float = 0.038,
) -> Dict[str, float]:
    """Compute top-row KPIs with institutional definitions."""
    ret = _daily_ret_from_positions(positions)
    cum = _geom_cum(ret)
    ann_ret = _ann_return_12m(ret)
    vol12 = _ann_vol_12m(ret)
    sharpe = _sharpe_12m(ret, rf_annual=rf_annual)
    mdd = _max_drawdown_from_cum(cum)

    # AUM now
    aum_now = float("nan")
    if not aum.empty and "AUM" in aum.columns:
        try:
            if "Date" in aum.columns:
                aum_now = float(aum.sort_values("Date")["AUM"].iloc[-1])
            else:
                aum_now = float(aum["AUM"].iloc[-1])
        except:
            aum_now = float("nan")

    # Liquidity <=90d
    liq_col = None
    for c in ["DAYS_TO_LIQUIDATE", "DaysToLiq", "DaysToLiquidate"]:
        if c in positions.columns:
            liq_col = c
            break
    
    if liq_col:
        w = positions.groupby(["Date"]).apply(
            lambda x: np.average((x[liq_col] <= 90).astype(float), weights=np.abs(x.get("GMV", 1.0)))
        )
        liq_90 = float(w.iloc[-1])
    else:
        liq_90 = float("nan")

    return dict(
        ann_return_12m=ann_ret,
        vol_12m=vol12,
        sharpe_12m=sharpe,
        cum_return_itd=float(cum.iloc[-1]) if len(cum) else float("nan"),
        max_dd_itd=mdd,
        aum_now=aum_now,
        liq_90d_pct=liq_90,
    )

def perf_timeseries_vs_bench(
    positions: pd.DataFrame, bench: pd.DataFrame, benchmark_col: str
) -> pd.DataFrame:
    """Returns DataFrame with Date, FundCum, BenchCum for cumulative returns."""
    ret = _daily_ret_from_positions(positions)
    cum_fund = _geom_cum(ret).rename("FundCum").to_frame()
    b = bench[["Date", benchmark_col]].copy()
    b = b.dropna().set_index("Date")[benchmark_col].astype(float)
    cum_bench = _geom_cum(b).rename("BenchCum").to_frame()
    out = cum_fund.join(cum_bench, how="inner").reset_index().rename(columns={"index":"Date"})
    return out

def rolling_excess_return(positions: pd.DataFrame, bench: pd.DataFrame, window: int = 90, benchmark_col: Optional[str] = None) -> pd.DataFrame:
    """Compute rolling excess return vs benchmark.
    
    Args:
        positions: Portfolio positions data
        bench: Benchmark data
        window: Rolling window size in days
        benchmark_col: Name of the benchmark column to use (if None, uses first numeric column)
    """
    ret_series = _daily_ret_from_positions(positions)
    
    if bench.empty or len(ret_series) == 0:
        return pd.DataFrame(columns=["Date", "ExcessReturn"])
    
    # Convert Series to DataFrame
    if isinstance(ret_series, pd.Series):
        ret = ret_series.reset_index()
        ret.columns = ["Date", "Ret"]
    else:
        ret = ret_series.reset_index()
    
    # Get benchmark returns if available
    if "Date" in bench.columns and len(bench) > 0:
        # Look for return columns (RET in name, or Return, or numeric columns like ACWI, SPY)
        numeric_cols = bench.select_dtypes(include=[np.number]).columns.tolist()
        # Remove Date column if it was included
        if "Date" in numeric_cols:
            numeric_cols.remove("Date")
        
        if numeric_cols:
            # Use specified benchmark column if provided and exists, otherwise use first numeric column
            if benchmark_col and benchmark_col in numeric_cols:
                bench_col = benchmark_col
            else:
                bench_col = numeric_cols[0]
            bench_daily = bench.groupby("Date", observed=True)[bench_col].mean().reset_index()
            bench_daily.columns = ["Date", "BenchRet"]
            ret = ret.merge(bench_daily, on="Date", how="inner")
            ret["ExcessReturn"] = ret["Ret"] - ret["BenchRet"]
    
    if "ExcessReturn" not in ret.columns:
        ret["ExcessReturn"] = 0
    
    # Rolling window (convert to basis points for display)
    ret["RollingExcess"] = ret["ExcessReturn"].rolling(window=min(window, len(ret))).mean() * 10000
    
    return ret[["Date", "ExcessReturn", "RollingExcess"]]

def account_annualized_returns(positions: pd.DataFrame) -> pd.DataFrame:
    """Annualized returns by account."""
    if "Account" not in positions.columns or "RET_TWR" not in positions.columns:
        return pd.DataFrame(columns=["Account", "AnnReturn"])
    
    account_rets = positions.groupby("Account", observed=True)["RET_TWR"].agg([
        lambda x: x.mean() * 252,
        lambda x: x.std() * np.sqrt(252) if len(x) > 1 else 0,
        "count"
    ]).reset_index()
    account_rets.columns = ["Account", "AnnReturn", "AnnVol", "Observations"]
    
    return account_rets[["Account", "AnnReturn", "AnnVol"]]

def tracking_error_and_info_ratio(positions: pd.DataFrame, bench: pd.DataFrame, benchmark_col: Optional[str] = None) -> Dict[str, float]:
    """Compute tracking error and information ratio vs benchmark.
    
    Args:
        positions: Portfolio positions data
        bench: Benchmark data
        benchmark_col: Name of the benchmark column to use (if None, uses first numeric column)
    """
    if bench.empty or "RET_TWR" not in positions.columns:
        return dict(tracking_error=float("nan"), info_ratio=float("nan"), beta=float("nan"), alpha=float("nan"))
    
    ret_series = _daily_ret_from_positions(positions)
    
    # Convert Series to DataFrame
    if isinstance(ret_series, pd.Series):
        ret = ret_series.reset_index()
        ret.columns = ["Date", "Ret"]
    else:
        ret = ret_series.reset_index()
    
    # Get benchmark returns - look for numeric columns (ACWI, SPY, etc.)
    if "Date" in bench.columns and len(bench) > 0:
        numeric_cols = bench.select_dtypes(include=[np.number]).columns.tolist()
        if "Date" in numeric_cols:
            numeric_cols.remove("Date")
        
        if numeric_cols:
            # Use specified benchmark column if provided and exists, otherwise use first numeric column
            if benchmark_col and benchmark_col in numeric_cols:
                bench_col = benchmark_col
            else:
                bench_col = numeric_cols[0]  # Use first numeric column (e.g., ACWI)
            bench_daily = bench.groupby("Date", observed=True)[bench_col].mean().reset_index()
            bench_daily.columns = ["Date", "BenchRet"]
            ret = ret.merge(bench_daily, on="Date", how="inner")
            
            if "BenchRet" in ret.columns:
                # Drop rows with NaN values for clean calculations
                ret_clean = ret[~ret["Ret"].isna() & ~ret["BenchRet"].isna()].copy()
                
                if len(ret_clean) > 1:
                    ret_clean["ExcessReturn"] = ret_clean["Ret"] - ret_clean["BenchRet"]
                    
                    # Tracking error (annualized std of excess returns)
                    te = ret_clean["ExcessReturn"].std() * np.sqrt(252)
                    
                    # Beta = Cov(portfolio, benchmark) / Var(benchmark)
                    cov_matrix = np.cov(ret_clean["Ret"], ret_clean["BenchRet"])
                    beta = cov_matrix[0, 1] / ret_clean["BenchRet"].var() if ret_clean["BenchRet"].var() > 0 else float("nan")
                    
                    # Alpha = Portfolio return - (risk-free rate + beta * (benchmark return - risk-free rate))
                    # Using risk-free rate = 3.8%
                    bench_ann = ret_clean["BenchRet"].mean() * 252
                    port_ann = ret_clean["Ret"].mean() * 252
                    alpha = port_ann - (0.038 + beta * (bench_ann - 0.038)) if not np.isnan(beta) else float("nan")
                    
                    # Information ratio = excess return / tracking error
                    excess_ret_ann = ret_clean["ExcessReturn"].mean() * 252
                    ir = excess_ret_ann / te if te > 0 else float("nan")
                    
                    return dict(
                        tracking_error=float(te),
                        info_ratio=float(ir),
                        beta=float(beta),
                        alpha=float(alpha)
                    )
    
    return dict(tracking_error=float("nan"), info_ratio=float("nan"), beta=float("nan"), alpha=float("nan"))

def perf_account_kpis(positions: pd.DataFrame) -> pd.DataFrame:
    """Return per-account KPIs table."""
    req = []
    for acct, df in positions.groupby("Account", observed=True):
        ret = _daily_ret_from_positions(df)
        cum = _geom_cum(ret)
        row = dict(
            Account=acct,
            CumRet=float(cum.iloc[-1]) if len(cum) else np.nan,
            AnnReturn12M=_ann_return_12m(ret),
            AnnVol12M=_ann_vol_12m(ret),
            Sharpe12M=_sharpe_12m(ret),
            MaxDrawdown=_max_drawdown_from_cum(cum),
            Obs=int(len(ret)),
        )
        req.append(row)
    return pd.DataFrame(req).sort_values("AnnReturn12M", ascending=False, na_position="last")

# ============================================================================
# Liquidity Functions
# ============================================================================

def liquidity_kpis(positions: pd.DataFrame) -> Dict[str, float]:
    """Compute liquidity KPIs for top row display."""
    df = positions.copy()
    
    # Weighted ADV Cover
    if "ADV_COVER_GMV" in df.columns and "GMV" in df.columns:
        latest = df[df["Date"] == df["Date"].max()]
        weighted_adv = (latest["GMV"].abs() * latest["ADV_COVER_GMV"]).sum() / latest["GMV"].abs().sum()
    else:
        weighted_adv = float("nan")
    
    # % GMV Unwindable <5d
    if "ADV_COVER_GMV" in df.columns and "GMV" in df.columns:
        latest = df[df["Date"] == df["Date"].max()]
        unwind_gmv = latest[latest["ADV_COVER_GMV"] <= 5]["GMV"].abs().sum()
        total_gmv = latest["GMV"].abs().sum()
        pct_unwind = unwind_gmv / total_gmv if total_gmv > 0 else float("nan")
    else:
        pct_unwind = float("nan")
    
    # % GMV Illiquid >20d
    if "ADV_COVER_GMV" in df.columns and "GMV" in df.columns:
        latest = df[df["Date"] == df["Date"].max()]
        illiquid_gmv = latest[latest["ADV_COVER_GMV"] > 20]["GMV"].abs().sum()
        total_gmv = latest["GMV"].abs().sum()
        pct_illiquid = illiquid_gmv / total_gmv if total_gmv > 0 else float("nan")
    else:
        pct_illiquid = float("nan")
    
    # Liquidity trend (↑/↓)
    if "ADV_COVER_GMV" in df.columns and "Date" in df.columns:
        df_sorted = df.sort_values("Date")
        latest_mean = df_sorted[df_sorted["Date"] >= df_sorted["Date"].max() - pd.Timedelta(days=30)]["ADV_COVER_GMV"].mean()
        prev_mean = df_sorted[(df_sorted["Date"] >= df_sorted["Date"].max() - pd.Timedelta(days=60)) & 
                              (df_sorted["Date"] < df_sorted["Date"].max() - pd.Timedelta(days=30))]["ADV_COVER_GMV"].mean()
        trend = "↑" if latest_mean < prev_mean else "↓" if latest_mean > prev_mean else "→"
    else:
        trend = "N/A"
    
    return dict(
        weighted_adv_cover=weighted_adv,
        pct_unwind_lt5d=pct_unwind,
        pct_illiquid_gt20d=pct_illiquid,
        liquidity_trend=trend
    )

def liquidity_buckets(positions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Map positions to liquidity buckets using ADV_COVER_GMV."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()]
    
    if "ADV_COVER_GMV" in df.columns and "GMV" in df.columns:
        # Create buckets based on ADV_COVER_GMV
        days = latest["ADV_COVER_GMV"].fillna(9999)
        bucket = pd.cut(days, bins=[-1, 1, 5, 20, 99999], 
                       labels=["≤1d", "1–5d", "5–20d", ">20d"])
        
        agg = latest.groupby(bucket, observed=True).apply(lambda x: x["GMV"].abs().sum()).to_frame()
        agg.columns = ["GMV_Sum"]
        agg["Pct"] = agg["GMV_Sum"] / agg["GMV_Sum"].sum()
        result = agg.reset_index()
        result.columns = ["Bucket", "GMV_Sum", "Pct"]
        return result
    else:
        return pd.DataFrame(columns=["Bucket", "GMV_Sum", "Pct"])

def liquidity_trend_over_time(positions: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted ADV_COVER_GMV over time."""
    df = positions.copy()
    
    if "ADV_COVER_GMV" in df.columns and "GMV" in df.columns and "Date" in df.columns:
        daily_liq = df.groupby("Date").apply(
            lambda x: np.average(x["ADV_COVER_GMV"].fillna(9999), weights=x["GMV"].abs())
        ).reset_index()
        daily_liq.columns = ["Date", "Weighted_ADV_Cover"]
        return daily_liq
    else:
        return pd.DataFrame(columns=["Date", "Weighted_ADV_Cover"])

def aum_stack(aum: pd.DataFrame) -> pd.DataFrame:
    """AUM stack data."""
    return aum.copy()

def flow_waterfall(aum: pd.DataFrame) -> pd.DataFrame:
    """Sub/Red flow waterfall."""
    cols = [c for c in ["Subscriptions", "Redemptions", "NetFlow"] if c in aum.columns]
    return aum[["Date"] + cols].copy() if cols else pd.DataFrame(columns=["Date", "Subscriptions", "Redemptions", "NetFlow"])

def account_terms_table(accounts: pd.DataFrame) -> pd.DataFrame:
    """Account terms table."""
    return accounts.copy()

# ============================================================================
# Risk & Attribution Functions
# ============================================================================

# --- Risk Exposure Functions ---

def risk_exposure_kpis(positions: pd.DataFrame) -> Dict[str, float]:
    """Compute risk exposure KPIs: Top Region, Top Sector, Gross Leverage, Net Exposure."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    kpis = {}
    
    # Top Region and Sector by exposure
    if "Region" in df.columns and "NMV" in df.columns:
        region_exp = latest.groupby("Region").apply(lambda x: x["NMV"].abs().sum())
        kpis["top_region"] = region_exp.idxmax() if not region_exp.empty else "N/A"
        kpis["top_region_pct"] = (region_exp.max() / latest["NMV"].abs().sum() * 100) if not region_exp.empty else 0
    else:
        kpis["top_region"] = "N/A"
        kpis["top_region_pct"] = 0
    
    if "GICS_SECTOR_CLEAN" in df.columns and "NMV" in df.columns:
        sector_exp = latest.groupby("GICS_SECTOR_CLEAN").apply(lambda x: x["NMV"].abs().sum())
        kpis["top_sector"] = sector_exp.idxmax() if not sector_exp.empty else "N/A"
        kpis["top_sector_pct"] = (sector_exp.max() / latest["NMV"].abs().sum() * 100) if not sector_exp.empty else 0
    else:
        kpis["top_sector"] = "N/A"
        kpis["top_sector_pct"] = 0
    
    # Leverage metrics - calculate from GMV and NMV
    if "GMV" in latest.columns and "NMV" in latest.columns:
        total_gmv = latest["GMV"].abs().sum()
        total_nmv = latest["NMV"].abs().sum()
        kpis["gross_leverage"] = (total_gmv / total_nmv) if total_nmv > 0 else 0.0
        
        # Net exposure = NMV / NAV
        sum_nmv = latest["NMV"].sum()
        kpis["net_exposure"] = (sum_nmv / total_nmv) if total_nmv > 0 else 0.0
    elif "LEVERAGE_FUND" in latest.columns:
        kpis["gross_leverage"] = float(latest["LEVERAGE_FUND"].iloc[-1]) if len(latest) > 0 else 0.0
        kpis["net_exposure"] = float(latest["LEVERAGE_FUND"].iloc[-1]) if len(latest) > 0 else 0.0
    else:
        kpis["gross_leverage"] = 0.0
        kpis["net_exposure"] = 0.0
    
    return kpis

def top_sector_exposures(positions: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Top N sector exposures sorted by absolute exposure, with net direction shown."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if "GICS_SECTOR_CLEAN" not in df.columns or "NMV" not in df.columns:
        return pd.DataFrame(columns=["Sector", "Net_NMV", "Abs_NMV", "Pct"])
    
    # Calculate net exposure per sector
    sector_nmv_net = latest.groupby("GICS_SECTOR_CLEAN", observed=True)["NMV"].sum()
    # Calculate absolute exposure per sector
    sector_nmv_abs = latest.groupby("GICS_SECTOR_CLEAN", observed=True)["NMV"].apply(lambda x: x.abs().sum())
    
    # Sort by absolute exposure, take top N
    sector_nmv_abs_sorted = sector_nmv_abs.nlargest(top_n)
    
    # Get corresponding net values for display
    result = pd.DataFrame({
        "Sector": sector_nmv_abs_sorted.index,
        "Net_NMV": [sector_nmv_net[sect] for sect in sector_nmv_abs_sorted.index],
        "Abs_NMV": sector_nmv_abs_sorted.values
    })
    
    # Calculate % of total NAV (using absolute)
    total_nav = latest["NMV"].abs().sum()
    result["Pct"] = result["Abs_NMV"] / total_nav * 100
    
    return result[["Sector", "Net_NMV", "Abs_NMV", "Pct"]]

def regional_exposure_pie(positions: pd.DataFrame) -> pd.DataFrame:
    """Regional exposure as % of NAV."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if "Region" not in df.columns or "NMV" not in df.columns:
        return pd.DataFrame(columns=["Region", "Pct"])
    
    region_nmv = latest.groupby("Region", observed=True)["NMV"].sum()
    total_nav = latest["NMV"].abs().sum()
    
    result = pd.DataFrame({
        "Region": region_nmv.index,
        "Pct": (region_nmv / total_nav * 100).values
    })
    
    return result

def market_cap_classify(market_cap: float) -> str:
    """Classify market cap into categories."""
    if pd.isna(market_cap) or market_cap <= 0:
        return "Unknown"
    elif market_cap >= 200_000_000_000:  # $200B+
        return "Mega Cap"
    elif market_cap >= 10_000_000_000:  # $10B+
        return "Large Cap"
    elif market_cap >= 2_000_000_000:  # $2B+
        return "Mid Cap"
    elif market_cap >= 300_000_000:  # $300M+
        return "Small Cap"
    else:
        return "Micro Cap"

def market_cap_exposure_pie(positions: pd.DataFrame) -> pd.DataFrame:
    """Market cap exposure as % of NAV."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if "MARKET_CAP" not in df.columns or "NMV" not in df.columns:
        return pd.DataFrame(columns=["MarketCap", "Pct"])
    
    # Classify each position's market cap
    latest = latest.copy()
    latest["MarketCap_Bucket"] = latest["MARKET_CAP"].apply(market_cap_classify)
    
    # Use absolute value for each bucket to show true exposure weight
    cap_nmv_abs = latest.groupby("MarketCap_Bucket", observed=True)["NMV"].apply(lambda x: x.abs().sum())
    total_nav = latest["NMV"].abs().sum()
    
    result = pd.DataFrame({
        "MarketCap": cap_nmv_abs.index,
        "Pct": (cap_nmv_abs / total_nav * 100).values
    })
    
    return result

def sector_exposure_pie(positions: pd.DataFrame) -> pd.DataFrame:
    """Sector exposure as % of NAV (using absolute values to sum to 100%)."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if "GICS_SECTOR_CLEAN" not in df.columns or "NMV" not in df.columns:
        return pd.DataFrame(columns=["Sector", "Pct"])
    
    # Use absolute value for each sector to show true exposure weight
    sector_nmv_abs = latest.groupby("GICS_SECTOR_CLEAN", observed=True)["NMV"].apply(lambda x: x.abs().sum())
    total_nav = latest["NMV"].abs().sum()
    
    # Calculate sign to show long/short
    sector_nmv_net = latest.groupby("GICS_SECTOR_CLEAN", observed=True)["NMV"].sum()
    
    result = pd.DataFrame({
        "Sector": sector_nmv_abs.index,
        "Pct": (sector_nmv_abs / total_nav * 100).values,
        "Net_NMV": sector_nmv_net.values
    })
    
    # Add sign indicator in the label
    result["Sector_Label"] = result.apply(
        lambda row: f"{row['Sector']} ({'Long' if row['Net_NMV'] > 0 else 'Short' if row['Net_NMV'] < 0 else 'Flat'})",
        axis=1
    )
    
    return result[["Sector", "Pct", "Sector_Label"]]

def leverage_trend(positions: pd.DataFrame) -> pd.DataFrame:
    """Fund leverage trend over time.
    
    Calculates: Leverage = GMV / abs(NMV)
    This gives fund-level gross leverage over time.
    """
    df = positions.copy()
    if "Date" not in df.columns:
        return pd.DataFrame(columns=["Date", "Leverage"])
    
    if "GMV" not in df.columns or "NMV" not in df.columns:
        return pd.DataFrame(columns=["Date", "Leverage"])
    
    # Calculate leverage per day
    daily = df.groupby("Date", observed=True).agg({
        "GMV": "sum",
        "NMV": lambda x: x.abs().sum()
    }).reset_index()
    
    daily["Leverage"] = daily["GMV"] / daily["NMV"].clip(lower=1)
    
    return daily[["Date", "Leverage"]]


# --- Return Attribution Functions ---

def top_contributors(positions: pd.DataFrame, reference: pd.DataFrame, by_dimension: str = "GICS_SECTOR_CLEAN", top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Top positive and negative contributors by dimension."""
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if by_dimension not in df.columns or "PNL" not in df.columns:
        return pd.DataFrame(columns=[by_dimension, "PNL"]), pd.DataFrame(columns=[by_dimension, "PNL"])
    
    contrib = latest.groupby(by_dimension, observed=True)["PNL"].sum()
    contrib = contrib.sort_values()
    
    top_neg = contrib.head(top_n).reset_index()
    top_neg.columns = ["Bucket", "Contribution"]
    
    top_pos = contrib.tail(top_n).reset_index()
    top_pos.columns = ["Bucket", "Contribution"]
    
    return top_neg, top_pos

def return_by_dimension(positions: pd.DataFrame, by_dimension: str) -> pd.DataFrame:
    """Calculate return by dimension using PNL/NMV method to match attribution table.
    
    Formula: Return (bps) = (sum(PNL) / sum(abs(NMV))) * 10000
    This matches the calculation in the Attribution Details table.
    """
    df = positions.copy()
    latest = df[df["Date"] == df["Date"].max()] if "Date" in df.columns else df
    
    if by_dimension not in df.columns:
        return pd.DataFrame(columns=[by_dimension, "Return"])
    
    # Calculate return using PNL and NMV (consistent with attribution table)
    # Formula: (sum(PNL) / sum(abs(NMV))) * 10000 for bps
    if "PNL" in latest.columns and "NMV" in latest.columns:
        grouped = latest.groupby(by_dimension, observed=True).agg({
            "PNL": "sum",
            "NMV": lambda x: x.abs().sum()
        }).reset_index()
        
        # Handle division by zero
        grouped["Return"] = ((grouped["PNL"] / grouped["NMV"].clip(lower=1e-6)) * 10000).round(2)
        
        # Reorder columns
        dim_col_name = list(grouped.columns)[0]
        result = grouped[[dim_col_name, "Return"]].copy()
        result.columns = [by_dimension, "Return"]
        
        # Sort by Return
        result = result.sort_values("Return")
    else:
        return pd.DataFrame(columns=[by_dimension, "Return"])
    
    return result

def simplified_brinson(positions: pd.DataFrame, bench: pd.DataFrame, by_dimension: str = "GICS_SECTOR_CLEAN") -> pd.DataFrame:
    """Simplified Brinson decomposition (Allocation + Selection)."""
    # Stub - would require benchmark weights and returns
    return pd.DataFrame(columns=["Dimension", "Bucket", "Port_Wt", "Bench_Wt", "Return", "Attribution_bps"])

def exposures_heatmap(positions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Exposure heatmap stub."""
    return pd.DataFrame()

def factor_attribution(positions: pd.DataFrame, bench: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Factor attribution stub."""
    return pd.DataFrame(), pd.DataFrame()

def scenario_pnl(positions: pd.DataFrame, shocks: Dict[str, float]) -> pd.DataFrame:
    """Scenario PnL stub."""
    return pd.DataFrame()

# ============================================================================
# QA Functions
# ============================================================================

def qa_checks(positions: pd.DataFrame, reference: pd.DataFrame, aum: pd.DataFrame) -> List[Dict[str, str]]:
    """Return QA issues list."""
    issues = []
    
    if "SEDOL" in reference.columns and "USym" in reference.columns:
        dup = reference.groupby("SEDOL")["USym"].nunique(dropna=True)
        n = int((dup > 1).sum())
        if n > 0:
            issues.append(dict(severity="Medium", type="Duplicate Mapping", count=n, description=f"{n} SEDOLs map to multiple USym"))
    
    try:
        r = _daily_ret_from_positions(positions)
        n_ext = int(((r > 1.0) | (r < -0.5)).sum())
        if n_ext > 0:
            issues.append(dict(severity="Low", type="Extreme Returns", count=n_ext, description=f"{n_ext} days with return >100% or <-50%"))
    except Exception:
        pass
    
    return issues

def qa_summary_text(positions: pd.DataFrame, reference: pd.DataFrame, aum: pd.DataFrame) -> str:
    """Produce QA summary text."""
    issues = qa_checks(positions, reference, aum)
    lines = [f"- {i['severity']}: {i['type']} — {i['count']} ({i['description']})" for i in issues]
    total_rows = len(positions)
    date_min = positions["Date"].min() if "Date" in positions.columns else None
    date_max = positions["Date"].max() if "Date" in positions.columns else None
    lines.append(f"- Data rows: {total_rows:,}")
    if date_min is not None and date_max is not None:
        lines.append(f"- Date range: {date_min.date()} → {date_max.date()}")
    return "\n".join(lines)

# ============================================================================
# Legacy Functions (for backwards compatibility)
# ============================================================================

def coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy coerce function."""
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ["PNL","GMV","NMV","LONG_GMV","SHORT_GMV","LONG_NMV","SHORT_NMV","PNL_CONTRIB_NAV","RC_NAV","ADV_COVER_GMV","DAYS_TO_LIQUIDATE","ADV_20D"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Date" in df.columns:
        df["Month"] = df["Date"].dt.to_period("M").astype(str)
        df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    return df

def aggregate_returns(df: pd.DataFrame, leaf_keys: Optional[List[str]] = None, target_keys: Optional[List[str]] = None, 
                     date_col: str = "Date", account_col: str = "Account", ret_col: str = "RET_TWR",
                     pnl_col: str = "PNL", nmv_col: str = "NMV", gmv_col: str = "GMV") -> pd.DataFrame:
    """Legacy aggregate returns."""
    df = df.copy()
    if leaf_keys is None:
        leaf_keys = [date_col, account_col] + ([c for c in ["SEDOL"] if c in df.columns])
    if target_keys is None:
        target_keys = [date_col, account_col]
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([*target_keys, *[k for k in leaf_keys if k not in target_keys]])

    leaf_entity_keys = [k for k in leaf_keys if k != date_col]
    df["Prev_NMV_leaf"] = df.groupby(leaf_entity_keys, dropna=False)[nmv_col].shift(1)

    agg_dict = {pnl_col: "sum", nmv_col: "sum"}
    if gmv_col and gmv_col in df.columns:
        agg_dict[gmv_col] = "sum"
    grouped = df.groupby(target_keys, dropna=False).agg(agg_dict).reset_index()

    prev = (df.groupby(target_keys, dropna=False)["Prev_NMV_leaf"]
            .sum(min_count=1).reset_index().rename(columns={"Prev_NMV_leaf": "Prev_NMV"}))
    out = grouped.merge(prev, on=target_keys, how="left")

    if out["Prev_NMV"].notna().any():
        out["Ret"] = out[pnl_col] / out["Prev_NMV"].replace({0: np.nan})
    else:
        if ret_col in df.columns:
            denom = df.groupby(target_keys, dropna=False)["Prev_NMV_leaf"].transform("sum")
            df["_w_ret"] = np.where(denom != 0, df["Prev_NMV_leaf"] / denom, np.nan) * df[ret_col]
            wret = df.groupby(target_keys, dropna=False)["_w_ret"].sum(min_count=1).reset_index()
            out = out.merge(wret, on=target_keys, how="left")
            out["Ret"] = out["_w_ret"]
            out = out.drop(columns=["_w_ret"])
        else:
            out["Ret"] = np.nan
    return out

def perf_stats(df: pd.DataFrame, by: Optional[List[str]] = None, date_col: str = "Date", ret_col: str = "Ret",
               ann: int = 252, date_start: Optional[str] = None, date_end: Optional[str] = None) -> pd.DataFrame:
    """Legacy perf stats."""
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    if date_start:
        d = d[d[date_col] >= pd.to_datetime(date_start)]
    if date_end:
        d = d[d[date_col] <= pd.to_datetime(date_end)]

    if ret_col not in d.columns and set(["PNL", "Prev_NMV"]).issubset(d.columns):
        d[ret_col] = d["PNL"] / d["Prev_NMV"].replace({0: np.nan})

    by = by or []
    groups = d.groupby(by, dropna=False) if by else [((), d)]
    rows = []
    iterator = groups if by else [((), d)]
    for key, g in iterator:
        r = g[ret_col].dropna() if ret_col in g.columns else pd.Series(dtype=float)
        if r.empty:
            rows.append({**(dict(zip(by, key)) if by else {}), "Obs": 0, "CumRet": np.nan, "AnnReturn_inferred": np.nan,
                         "Ret_mean_daily": np.nan, "Vol_daily": np.nan, "Sharpe_ann": np.nan, "MaxDrawdown": np.nan})
            continue
        mean = r.mean()
        vol = r.std(ddof=1)
        sharpe = (mean * ann) / (vol * np.sqrt(ann)) if vol and not np.isnan(vol) else np.nan
        eq = (1 + r).cumprod()
        peak = eq.cummax()
        mdd = (eq / peak - 1).min()
        cumret = eq.iloc[-1] - 1
        N = int(r.shape[0])
        annret = (1 + cumret) ** (ann / N) - 1 if N > 0 else np.nan
        rows.append({
            **(dict(zip(by, key)) if by else {}), "CumRet": cumret, "AnnReturn_inferred": annret,
            "Ret_mean_daily": mean, "Vol_daily": vol, "Sharpe_ann": sharpe, "MaxDrawdown": mdd, "Obs": N,})
    return pd.DataFrame(rows)

def liquidity_snapshot(df: pd.DataFrame, on: Optional[List[str]] = None) -> pd.DataFrame:
    """Legacy liquidity snapshot."""
    cols = [c for c in ["ADV_COVER_GMV", "DAYS_TO_LIQUIDATE", "GMV", "NMV"] if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=(on or []))
    if on is None:
        on = [c for c in ["Date", "Account"] if c in df.columns] or []
    g = df.groupby(on, dropna=False)[cols].agg(["mean", "median", "max", "sum"])
    g.columns = ["_".join(col) for col in g.columns]
    return g.reset_index()

def group_return_attrib(df: pd.DataFrame, dims: List[str]) -> pd.DataFrame:
    """Legacy group return attrib."""
    contrib_col = "PNL_CONTRIB_NAV" if "PNL_CONTRIB_NAV" in df.columns else "PNL"
    gcols = ["Date", "Account"] + [d for d in dims if d in df.columns]
    out = (df.groupby(gcols, dropna=False)[[contrib_col] + [c for c in ["GMV", "NMV"] if c in df.columns]].sum().reset_index())
    out = out.rename(columns={contrib_col: "Contrib"})
    return out

def top_n_contributors(df_grouped: pd.DataFrame, n: int = 10, sign: str = "both") -> pd.DataFrame:
    """Legacy top n contributors."""
    key = [c for c in ["Date", "Account"] if c in df_grouped.columns]
    if not key:
        d = df_grouped.copy()
        if sign == "pos":
            return d.sort_values("Contrib", ascending=False).head(n)
        if sign == "neg":
            return d.sort_values("Contrib", ascending=True).head(n)
        return pd.concat([d.sort_values("Contrib", ascending=False).head(n), d.sort_values("Contrib", ascending=True).head(n)], ignore_index=True)
    res = []
    for _, d in df_grouped.groupby(key, dropna=False):
        if sign == "pos":
            pick = d.sort_values("Contrib", ascending=False).head(n)
        elif sign == "neg":
            pick = d.sort_values("Contrib", ascending=True).head(n)
        else:
            pick = pd.concat([d.sort_values("Contrib", ascending=False).head(n), d.sort_values("Contrib", ascending=True).head(n)], ignore_index=True)
        res.append(pick)
    return pd.concat(res, ignore_index=True) if res else df_grouped.head(0)

def benchmark_comparison(account_returns: pd.DataFrame, benchmark_path: str, date_col: str = "Date", ret_col: str = "Ret", account_col: str = "Account") -> pd.DataFrame:
    """Legacy benchmark comparison."""
    # Implementation kept for compatibility
    try:
        bench = pd.read_csv(benchmark_path)
        if date_col in bench.columns:
            bench[date_col] = pd.to_datetime(bench[date_col], errors="coerce")
        for col in bench.columns:
            if col != date_col:
                bench[col] = pd.to_numeric(bench[col], errors='coerce')
    except Exception as e:
        return pd.DataFrame()

    account_returns[date_col] = pd.to_datetime(account_returns[date_col], errors="coerce")
    account_returns_clean = account_returns[[date_col, ret_col]].dropna(subset=[ret_col])
    merged = account_returns_clean.merge(bench.dropna(subset=[date_col]), on=date_col, how="inner")

    if merged.empty:
        return pd.DataFrame()
    bench_cols = [c for c in merged.columns if c not in [date_col, ret_col, account_col]]
    if not bench_cols:
        return pd.DataFrame()

    account_ret = merged[ret_col].dropna()
    account_ret_clipped = account_ret.clip(-0.50, 0.50)
    account_cum = np.nan
    if len(account_ret_clipped) > 0:
        cumprod = (1 + account_ret_clipped).cumprod()
        account_cum = cumprod.iloc[-1] - 1

    results = []
    for bench_col in bench_cols:
        bench_ret = merged[bench_col].dropna()
        if len(bench_ret) == 0:
            continue
        bench_ret_clipped = bench_ret.clip(-0.50, 0.50)
        bench_cum = np.nan
        if len(bench_ret_clipped) > 0:
            cumprod = (1 + bench_ret_clipped).cumprod()
            bench_cum = cumprod.iloc[-1] - 1
        ann_ret_account = np.nan
        ann_ret_bench = np.nan
        if len(account_ret_clipped) > 0:
            try:
                mean_daily = account_ret_clipped.mean()
                ann_ret_account = mean_daily * 252
            except:
                ann_ret_account = np.nan
        if len(bench_ret_clipped) > 0:
            try:
                mean_daily = bench_ret_clipped.mean()
                ann_ret_bench = mean_daily * 252
            except:
                ann_ret_bench = np.nan
        vol_account = account_ret_clipped.std() * np.sqrt(252) if len(account_ret_clipped) > 0 else np.nan
        vol_bench = bench_ret_clipped.std() * np.sqrt(252) if len(bench_ret_clipped) > 0 else np.nan
        sharpe_account = np.nan
        sharpe_bench = np.nan
        if vol_account > 0:
            sharpe_account = (account_ret_clipped.mean() * 252) / vol_account
        if vol_bench > 0:
            sharpe_bench = (bench_ret_clipped.mean() * 252) / vol_bench
        alpha = ann_ret_account - ann_ret_bench if not (np.isnan(ann_ret_account) or np.isnan(ann_ret_bench)) else np.nan
        results.append({
            "Benchmark": bench_col, "Account_CumReturn": account_cum, "Benchmark_CumReturn": bench_cum,
            "ExcessReturn": account_cum - bench_cum if not (np.isnan(account_cum) or np.isnan(bench_cum)) else np.nan,
            "Account_AnnReturn": ann_ret_account, "Benchmark_AnnReturn": ann_ret_bench,
            "Account_Volatility": vol_account, "Benchmark_Volatility": vol_bench,
            "Account_Sharpe": sharpe_account, "Benchmark_Sharpe": sharpe_bench, "Alpha": alpha,})

    return pd.DataFrame(results)

def prepare_benchmark_timeseries(account_returns: pd.DataFrame, benchmark_path: str, date_col: str = "Date", ret_col: str = "Ret") -> pd.DataFrame:
    """Legacy prepare benchmark timeseries."""
    try:
        bench = pd.read_csv(benchmark_path)
        if date_col in bench.columns:
            bench[date_col] = pd.to_datetime(bench[date_col], errors="coerce")
        for col in bench.columns:
            if col != date_col:
                bench[col] = pd.to_numeric(bench[col], errors='coerce')
    except Exception as e:
        return pd.DataFrame()

    account_df = account_returns[[date_col, ret_col]].copy()
    account_df[date_col] = pd.to_datetime(account_df[date_col], errors="coerce")
    account_df = account_df.dropna(subset=[date_col, ret_col]).sort_values(date_col)
    account_df[ret_col] = account_df[ret_col].clip(-0.50, 0.50)
    account_df["Account_CumReturn"] = (1 + account_df[ret_col]).cumprod() - 1

    merged = account_df[[date_col, "Account_CumReturn"]].merge(bench.dropna(subset=[date_col]), on=date_col, how="inner")
    if merged.empty:
        return pd.DataFrame()
    bench_cols = [c for c in merged.columns if c not in [date_col, "Account_CumReturn"]]
    for col in bench_cols:
        clipped_returns = merged[col].clip(-0.50, 0.50)
        merged[f"{col}_CumReturn"] = (1 + clipped_returns).cumprod() - 1
    return merged
