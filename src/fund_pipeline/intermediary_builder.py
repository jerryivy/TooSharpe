# fund_pipeline/intermediary_builder.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import shutil
import datetime as dt
import warnings


# =====================================
# Helpers (concise reST docstrings)
# =====================================

REQUIRED = ["Date", "Account", "SEDOL", "PNL", "GMV", "NMV"]

def _require_cols(df: pd.DataFrame, required: list[str]) -> None:
    """
    Validate that required columns exist in DataFrame.

    :param df: pandas.DataFrame
    :param required: list of required column names
    :raises KeyError: if any column missing
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _make_entity_id(df: pd.DataFrame, name: str = "ENTITY_ID", sep: str = "|") -> pd.DataFrame:
    """
    Create a primary key column as Account + SEDOL.

    :param df: pandas.DataFrame, input DataFrame with Account and SEDOL.
    :param name: str, column name for new entity id.
    :param sep: str, separator between Account and SEDOL.
    :return: pandas.DataFrame, with ENTITY_ID column added.
    """
    df = df.copy()
    if "Account" not in df.columns or "SEDOL" not in df.columns:
        raise KeyError("Missing required columns: 'Account' and/or 'SEDOL'.")
    df[name] = df["Account"].astype(str) + sep + df["SEDOL"].astype(str)
    return df


def _has_same_day_duplicates(df: pd.DataFrame) -> bool:
    """
    Check if (Date, Account, SEDOL) duplicates exist.

    :param df: pandas.DataFrame, input dataset.
    :return: bool, True if duplicates exist.
    """
    for c in ("Date", "Account", "SEDOL"):
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")
    return df.duplicated(["Date", "Account", "SEDOL"]).any()


def _compute_linking_metrics(
    df: pd.DataFrame,
    entity_keys: List[str] = ["Account", "SEDOL"],
    flow_eps: float = 1e-8,
    widx_base: float = 1.0,
    fund_aum_anchor: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute FLOW, RET_TWR, W_IDX, and leverage metrics.

    If ``fund_aum_anchor`` is provided, static notionals are scaled to the anchor.
    StaticNotional is account-level and repeated on each row; sums are computed
    over unique accounts to avoid double-counting.

    :param df: DataFrame with Date, Account, SEDOL, PNL, NMV, GMV, StaticNotional.
    :return: DataFrame with linking columns and leverage columns.
    """
    out = df.copy().sort_values(["Date"] + entity_keys).reset_index(drop=True)
    out.columns = out.columns.str.strip()

    # --- Linking ---
    out["NMV_LAG"] = out.groupby(entity_keys, dropna=False)["NMV"].shift(1)
    out["FLOW"] = out["NMV"] - (out["NMV_LAG"] + out["PNL"])
    out.loc[out["FLOW"].abs() < flow_eps, "FLOW"] = 0.0

    base = out["NMV_LAG"].where(out["NMV_LAG"] != 0, np.nan)
    out["RET_TWR"] = out["PNL"] / base
    out["W_IDX"] = out.groupby(entity_keys, dropna=False)["RET_TWR"].transform(
        lambda r: widx_base * (1.0 + r.fillna(0.0)).cumprod()
    )

    # --- Static notional hygiene (account-level, repeated on each row) ---
    if "StaticNotional" not in out.columns:
        raise KeyError("Column 'StaticNotional' is required in df.")
    out["StaticNotional"] = (
        out["StaticNotional"].astype(str).str.replace(r"[\$,]", "", regex=True).astype(float)
    )

    # Unique-accounts total notional (avoid double-counting across rows/dates/SEDOLs)
    acct_static = (
        out[["Account", "StaticNotional"]]
        .dropna(subset=["StaticNotional"])
        .drop_duplicates(subset=["Account"])
    )
    sum_static_unique = float(acct_static["StaticNotional"].sum())

    # Scaling factor and fund denominator
    if fund_aum_anchor is not None:
        k = float(fund_aum_anchor) / max(sum_static_unique, 1e-9)
        fund_denom = float(fund_aum_anchor)
    else:
        k = 1.0
        fund_denom = sum_static_unique

    # Per-row effective equity uses account-level notional × k
    out["EffectiveEquity"] = np.clip(out["StaticNotional"] * k, 1e-9, None)

    # --- Leverage ---
    by_acct = ["Date", "Account"]
    gmv_acct = out.groupby(by_acct, dropna=False)["GMV"].transform("sum")
    nmv_acct = out.groupby(by_acct, dropna=False)["NMV"].transform("sum")
    gmv_fund = out.groupby(["Date"], dropna=False)["GMV"].transform("sum")

    out["LEVERAGE_ACCT"] = gmv_acct / out["EffectiveEquity"]
    out["LEV_NET_ACCT"]  = nmv_acct.abs() / out["EffectiveEquity"]
    out["LEVERAGE_FUND"] = gmv_fund / max(fund_denom, 1e-9)

    return out


def _compute_cross_sectional_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute W_NAV, W_GROSS, W_NETABS.

    :param df: pandas.DataFrame, input dataset.
    :return: pandas.DataFrame, with weight columns.
    """
    df = df.copy()
    by_fund = ["Date", "Account"]
    sum_nmv = df.groupby(by_fund, dropna=False)["NMV"].transform("sum")
    sum_gmv = df.groupby(by_fund, dropna=False)["GMV"].transform("sum")
    sum_abs_nmv = df.groupby(by_fund, dropna=False)["NMV"].transform(lambda s: s.abs().sum())
    df["W_NAV"] = np.where(sum_nmv != 0, df["NMV"] / sum_nmv, np.nan)
    df["W_GROSS"] = np.where(sum_gmv != 0, df["GMV"] / sum_gmv, np.nan)
    df["W_NETABS"] = np.where(sum_abs_nmv != 0, df["NMV"] / sum_abs_nmv, np.nan)
    
    return df


def _compute_return_contributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RC_NAV and PNL_CONTRIB_NAV.

    :param df: pandas.DataFrame, input dataset.
    :return: pandas.DataFrame, with contribution columns.
    """
    df = df.copy()
    by_account = ["Date", "Account"]

    # previous period NAV (to calculate PNL_CONTRIB_NAV）
    fund_nav_lag = df.groupby(by_account, dropna=False)["NMV"].shift(1)

    # previous period W_NAV
    w_nav_lag = df.groupby(by_account, dropna=False)["W_NAV"].shift(1)

    # Return-based contribution（previous weights × current period TWR）
    df["RC_NAV"] = w_nav_lag * df["RET_TWR"]

    # PnL-based contribution（current PnL / previous NAV）
    df["PNL_CONTRIB_NAV"] = np.where(
        fund_nav_lag.notna() & (fund_nav_lag != 0),
        df["PNL"] / fund_nav_lag,
        np.nan,
    )
    return df

    
def _compute_long_short_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute long/short splits and SIGN_NMV for exposure analysis.
    Works for both long-only and long/short portfolios.

    Columns created:
        LONG_GMV   : gross market value of long positions
        SHORT_GMV  : gross market value of short positions
        LONG_NMV   : net market value (positive side)
        SHORT_NMV  : absolute net market value (negative side)
        SIGN_NMV   : +1 / -1 / 0 sign flag for NMV direction
    """
    df = df.copy()

    # --- GMV directional split ---
    df["LONG_GMV"] = np.where(df["NMV"] > 0, df["GMV"], 0.0)
    df["SHORT_GMV"] = np.where(df["NMV"] < 0, df["GMV"], 0.0)

    # --- NMV directional split ---
    df["LONG_NMV"] = np.where(df["NMV"] > 0, df["NMV"], 0.0)
    df["SHORT_NMV"] = np.where(df["NMV"] < 0, -df["NMV"], 0.0)

    # --- Directional flag ---
    df["SIGN_NMV"] = np.sign(df["NMV"])

    return df
 

def _liquidity_helpers(df: pd.DataFrame, participation_rate: float = 0.2) -> pd.DataFrame:
    """
    Compute ADV_COVER_GMV and DAYS_TO_LIQUIDATE.

    :param df: pandas.DataFrame, input dataset with ADV (dollar-based).
    :param participation_rate: float, tradable ADV fraction (e.g. 0.2 = 20%).
    :return: pandas.DataFrame, with liquidity columns.
    """
    df = df.copy()
    if "AverageDailyVolume_20_DAY" in df.columns:
        df["ADV_20D"] = pd.to_numeric(df["AverageDailyVolume_20_DAY"], errors="coerce")
        valid = df["ADV_20D"].notna() & (df["ADV_20D"] > 0)
        df["ADV_COVER_GMV"] = np.where(valid, df["GMV"] / df["ADV_20D"], np.nan)
        df["DAYS_TO_LIQUIDATE"] = np.where(
            valid, df["GMV"] / (df["ADV_20D"] * participation_rate), np.nan
        )
    return df

# ====== orchestrator (replace your current one) ======
def build_intermediary_from_integrated(
    df: pd.DataFrame,
    flow_eps: float = 1e-8,
    widx_base: float = 1.0,
    participation_rate: float = 0.2,
    do_same_day_aggregate: bool = False,   # 👈 默认关闭；你上游已处理
    return_diagnostics: bool = False,
    # --- save controls ---
    auto_save: bool = True,
    out_root: str | Path = "outputs",
    subdir: str = "intermediary",
    write_csv: bool = False,
    latest_link: bool = True,
    stamp: str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Build intermediary dataset at (Date, Account, SEDOL) and optionally save to disk.

    :param df (pd.DataFrame): Integrated security-level input with at least required columns.
    :param flow_eps (float): Epsilon threshold to zero-out tiny FLOW.
    :param widx_base (float): Base for wealth index linking.
    :param participation_rate (float): ADV participation rate for liquidity calc.
    :param do_same_day_aggregate (bool): If True, collapse same-day duplicates (usually not needed if upstream handled).
    :param return_diagnostics (bool): If True, return (df, diagnostics).
    :param auto_save (bool): Save to <out_root>/<subdir>/<timestamp>/ and mirror to latest/.
    :param out_root (str | Path): Root outputs folder (use "../outputs" when running from notebooks/).
    :param subdir (str): Subfolder under out_root.
    :param write_csv (bool): Also write CSV alongside Parquet.
    :param latest_link (bool): Mirror a copy to <out_root>/<subdir>/latest/.
    :param stamp (str | None): Optional timestamp; default current "YYYYMMDD_HHMMSS".
    :return (pd.DataFrame | (pd.DataFrame, dict)): Intermediary (and diagnostics if requested).
    """
    _require_cols(df, REQUIRED)
    diagnostics: dict = {}

    # 0) same-day collapse (usually OFF because upstream already did it)
    collapsed = 0
    if do_same_day_aggregate:
        if "_same_day_aggregate_if_needed" in globals():
            df, collapsed = _same_day_aggregate_if_needed(df)  # type: ignore[name-defined]
        else:
            warnings.warn(
                "do_same_day_aggregate=True but _same_day_aggregate_if_needed is not defined. Skipping collapse."
            )
    diagnostics["collapsed_rows"] = int(collapsed)
    diagnostics["did_aggregate"] = bool(collapsed > 0)

    # 1) core metrics pipeline
    df = _make_entity_id(df, name="ENTITY_ID", sep="|")
    df = _compute_linking_metrics(df, ["Account", "SEDOL"], flow_eps, widx_base)
    df = _compute_cross_sectional_weights(df)
    df = _compute_return_contributions(df)
    df = _compute_long_short_flags(df)
    df = _liquidity_helpers(df, participation_rate)
    diagnostics["rows"] = int(len(df))

    # 2) optional save
    if auto_save:
        save_info = _save_intermediary_payload(
            df,
            out_root=out_root,
            subdir=subdir,
            write_csv=write_csv,
            latest_link=latest_link,
            stamp=stamp,
        )
        diagnostics.update(save_info)

    return (df, diagnostics) if return_diagnostics else df


def _save_intermediary_payload(
    df_inter: pd.DataFrame,
    out_root: str | Path = "outputs",
    subdir: str = "intermediary",
    write_csv: bool = False,
    latest_link: bool = True,
    stamp: str | None = None,
) -> dict:
    """
    Persist intermediary to a timestamped bucket and update latest/.

    :param df_inter (pd.DataFrame): Intermediary to be saved.
    :param out_root (str | Path): Root outputs folder.
    :param subdir (str): Subfolder name under out_root.
    :param write_csv (bool): Also write CSV.
    :param latest_link (bool): Update latest/ mirror.
    :param stamp (str | None): Timestamp folder; if None, use current "YYYYMMDD_HHMMSS".
    :return (dict): Paths info: bucket, parquet, (csv), latest.
    """
    out_root = Path(out_root)
    stamp = stamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bucket = out_root / subdir / stamp
    bucket.mkdir(parents=True, exist_ok=True)

    pq_path = bucket / "intermediary.parquet"
    df_inter.to_parquet(pq_path, index=False)

    csv_path = None
    if write_csv:
        csv_path = bucket / "intermediary.csv"
        df_inter.to_csv(csv_path, index=False)

    latest = out_root / subdir / "latest"
    if latest_link:
        if latest.exists():
            shutil.rmtree(latest)
        latest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pq_path, latest / "intermediary.parquet")
        if csv_path:
            shutil.copy2(csv_path, latest / "intermediary.csv")

    info = {"bucket": str(bucket), "parquet": str(pq_path)}
    if csv_path:
        info["csv"] = str(csv_path)
    if latest_link:
        info["latest"] = str(latest)
    return info

