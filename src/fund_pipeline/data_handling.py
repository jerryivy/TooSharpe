"""
TooSharpe – Data Handling (Cleaning + Internal Integration)
-----------------------------------------------------------
What this script does:
1) Cleans each dataset independently (ReferenceData, Positions, Accounts, AUM, Index).
2) Integrates internal tables (Positions + Reference + Accounts).
3) Emits clean tables and a single analysis-ready integrated table.
4) DOES NOT merge benchmark returns at row-level; do that later at analysis time.

Usage (example):
    python toosharpe_data_handling.py \
        --raw-dir ./data \
        --out-dir ./outputs

Outputs:
    {out}/cleaned/*.csv
    {out}/integrated/TooSharpe_InternalIntegrated.csv
    {out}/manifest.json
"""

import os
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd


from fund_pipeline.utils import (
    _to_datetime_safe, save_csv, normalize_str_nulls, clean_categorical, impute_by_group_unique, ensure_dir
)

def drop_singleton_sedols(positions: pd.DataFrame, out_dir: str) -> tuple[pd.DataFrame, str | None]:
    """
    Identify and drop SEDOLs that appear only once in positions.
    Save dropped rows to outputs/dropped/positions_singleton_sedol.csv.
    Return (clean_positions, dropped_path or None).
    """
    pos = positions.copy()
    if "SEDOL" not in pos.columns:
        print("No 'SEDOL' column in positions — skip singleton check.")
        return pos, None

    pos["SEDOL"] = pos["SEDOL"].astype("string").str.strip()
    counts = pos["SEDOL"].value_counts(dropna=False)
    singletons = counts[counts == 1].index.tolist()

    if not singletons:
        print("No singleton SEDOLs detected — nothing dropped.")
        return pos, None

    dropped_dir = os.path.join(out_dir, "dropped")
    ensure_dir(dropped_dir)
    dropped_path = os.path.join(dropped_dir, "positions_singleton_sedol.csv")

    dropped_rows = pos[pos["SEDOL"].isin(singletons)].copy()
    save_csv(dropped_rows, dropped_path)

    cleaned = pos[~pos["SEDOL"].isin(singletons)].copy()
    print(f"Dropped {len(singletons)} singleton SEDOL(s) "
          f"({len(dropped_rows)} rows) → {dropped_path}")
    print("Reason: these SEDOLs appear only once in positions and are likely incomplete/anomalous.")

    return cleaned, dropped_path


def clean_reference_data(path: str, out_dir: str) -> pd.DataFrame:
    """
    Minimal & deterministic cleaner for ReferenceData.csv:
      1) Parse Date -> datetime and Month -> 'YYYY-MM' (string for stable joins)
      2) Standardize types: strings normalized, numerics coerced
      3) Impute GICS_SECTOR by neighbor uniqueness within ['SEDOL'] ONLY
      4) Split into GICS_SECTOR_CLEAN and Side (Short if endswith '-S', Long if non-empty)
      5) Keep a compact set of columns and drop exact duplicate rows only
    """

    # --- Load & basic columns ---
    ref = pd.read_csv(path)
    ref.columns = [c.strip() for c in ref.columns]

    # --- Dates ---
    ref["Date"] = _to_datetime_safe(ref.get("Date"))
    ref["Month"] = ref["Date"].dt.to_period("M").astype(str)  # string key for joins

    # --- String normalization (only the ones we need) ---
    if "SEDOL" in ref.columns:
        ref["SEDOL"] = normalize_str_nulls(ref["SEDOL"], to_na=True)
    if "GICS_SECTOR" in ref.columns:
        ref["GICS_SECTOR"] = normalize_str_nulls(ref["GICS_SECTOR"], to_na=True)

    # --- Numerics (coerce) ---
    for col in ["AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"]:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")

    # --- Impute GICS_SECTOR by neighbor uniqueness (SEDOL only) ---
    if {"SEDOL", "GICS_SECTOR"}.issubset(ref.columns):
        ref = impute_by_group_unique(
            ref, group_keys=["SEDOL"], target_cols=["GICS_SECTOR"], unknown="Unknown"
        )
        ref["GICS_SECTOR"] = clean_categorical(ref["GICS_SECTOR"], default="Unknown")

    # --- Split into CLEAN + Side ---
    ref["Side"] = ref["GICS_SECTOR"].apply(
        lambda x: "Short" if isinstance(x, str) and x.strip().endswith("-S")
        else "Long" if isinstance(x, str) and x.strip() != ""
        else "Unknown"
    )
    ref["GICS_SECTOR_CLEAN"] = (
        ref["GICS_SECTOR"].astype("string").str.replace("-S", "", regex=False).str.strip()
    )

    # --- ETF flag: ensure presence and integer form ---
    if "ETF_C" in ref.columns:
        ref["ETF_C"] = pd.to_numeric(ref["ETF_C"], errors="coerce").fillna(0).astype(int)
    else:
        ref["ETF_C"] = 0

    if "GICS_SECTOR" in ref.columns:
        ref = ref.drop(columns=["GICS_SECTOR"])
    ref = ref.drop_duplicates()

    # --- Save & return ---
    save_csv(ref, os.path.join(out_dir, "cleaned", "ReferenceData_clean.csv"))
    return ref


# def clean_positions(path: str, out_dir: str) -> pd.DataFrame:
#     pos = pd.read_csv(path)
#     pos.columns = [c.strip() for c in pos.columns]

#     # Dates
#     pos["Date"] = _to_datetime_safe(pos.get("Date"))
#     pos["Month"] = pos["Date"].dt.to_period("M").astype(str)

#     # Normalize core ID fields as strings
#     for id_col in ["Account", "SEDOL", "USym", "BBYKIdentifier", "BasicProduct", "Country", "Region"]:
#         if id_col in pos.columns:
#             pos[id_col] = normalize_str_nulls(pos[id_col], to_na=True)

#     # Numeric columns
#     for num_col in ["PNL", "GMV", "NMV"]:
#         if num_col in pos.columns:
#             pos[num_col] = pd.to_numeric(pos[num_col], errors="coerce")

#     # Impute categorical by neighbor uniqueness within (Account, SEDOL, USym)
#     group_keys = [k for k in ["Account", "SEDOL", "USym", "Country", "Region"] if k in pos.columns]
#     target_cols = [c for c in ["BasicProduct", "BBYKIdentifier"] if c in pos.columns]
#     if group_keys and target_cols:
#         pos = impute_by_group_unique(pos, group_keys=group_keys, target_cols=target_cols, unknown="Unknown")

#     pos = pos.drop_duplicates()
#     save_csv(pos, os.path.join(out_dir, "cleaned", "PositionLevelPNLAndExposure_clean.csv"))
#     return pos


def collapse_same_date_use_nextday_ids(
    df: pd.DataFrame,
    numeric_cols=("PNL","GMV","NMV"),
    id_cols=("USym","BasicProduct","BBYKIdentifier"),
    unknown_token="Unknown",
):
    """
    Collapse duplicate (Date, Account, SEDOL) rows by summing numeric columns
    and replacing identifier columns with their next-day values within the same
    (Account, SEDOL) entity.

    :param df: pandas.DataFrame, input dataset containing Date, Account, SEDOL.
    :param numeric_cols: tuple[str], numeric columns to sum when duplicates exist.
    :param id_cols: tuple[str], identifier columns to replace using next-day values.
    :param unknown_token: str, fallback string when next-day value is missing/NaN.
    :return: pandas.DataFrame, deduplicated dataset.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    keys = ["Date","Account","SEDOL"]
    entity = ["Account","SEDOL"]

    # 1) Aggregate duplicates: sum numeric columns, take last value for others
    use_num = [c for c in numeric_cols if c in df.columns]
    agg_map = {c: "sum" for c in use_num}
    for c in df.columns:
        if c not in set(keys + use_num):
            agg_map[c] = "last"
    out = (
        df.groupby(keys, dropna=False, as_index=False)
          .agg(agg_map)
          .sort_values(entity + ["Date"])
          .reset_index(drop=True)
    )

    # 2) Replace ID columns with next-day values within each (Account, SEDOL)
    g = out.groupby(entity, dropna=False)
    for col in id_cols:
        if col in out.columns:
            nxt = g[col].shift(-1)
            out[col] = nxt.combine_first(out[col]).fillna(unknown_token)

    return out.sort_values(keys).reset_index(drop=True)


def clean_positions(path: str, out_dir: str) -> pd.DataFrame:
    pos = pd.read_csv(path)
    pos.columns = [c.strip() for c in pos.columns]

    pos["Date"] = _to_datetime_safe(pos.get("Date"))
    pos["Month"] = pos["Date"].dt.to_period("M").astype(str)

    # normalize id columns
    for id_col in ["Account", "SEDOL", "USym", "BBYKIdentifier", "BasicProduct", "Country", "Region"]:
        if id_col in pos.columns:
            pos[id_col] = normalize_str_nulls(pos[id_col], to_na=True)

    # numeric coercion
    for num_col in ["PNL", "GMV", "NMV"]:
        if num_col in pos.columns:
            pos[num_col] = pd.to_numeric(pos[num_col], errors="coerce")

    # 🔹 drop and save singleton SEDOLs (your existing helper)
    pos, dropped_path = drop_singleton_sedols(pos, out_dir)

    # optional: impute BasicProduct / BBYKIdentifier by group before collapsing
    group_keys = [k for k in ["Account", "SEDOL", "USym", "Country", "Region"] if k in pos.columns]
    target_cols = [c for c in ["BasicProduct", "BBYKIdentifier"] if c in pos.columns]
    if group_keys and target_cols:
        pos = impute_by_group_unique(
            pos,
            group_keys=group_keys,
            target_cols=target_cols,
            unknown="Unknown"
        )

    # ✅ **改动 1：不再解包 (pos, collapsed)** —— 因为 collapse_same_date_use_nextday_ids() 只返回 df
    pos = pos.sort_values(["Account", "SEDOL", "Date"])
    pos = collapse_same_date_use_nextday_ids(
        pos,
        numeric_cols=("PNL", "GMV", "NMV"),
        id_cols=("USym", "BasicProduct", "BBYKIdentifier"),
        unknown_token="Unknown",
    )
    
    # ✅ **改动 2：确保输出目录存在** —— 避免第一次运行时因目录缺失报错
    os.makedirs(os.path.join(out_dir, "cleaned"), exist_ok=True)

    # Final dedupe and save
    pos = pos.drop_duplicates()
    save_csv(pos, os.path.join(out_dir, "cleaned", "PositionLevelPNLAndExposure_clean.csv"))

    # optional
    pos._dropped_singleton_path = dropped_path  # type: ignore[attr-defined]
    return pos


def clean_accounts(path: str, out_dir: str) -> pd.DataFrame:
    acct = pd.read_csv(path)
    acct.columns = [c.strip() for c in acct.columns]

    for col in ["StaticNotional", "DrawDownLimPctGMV", "DrawdownLimMultipleOfAnnualizedVol"]:
        if col in acct.columns:
            acct[col] = pd.to_numeric(acct[col], errors="coerce")

    for col in ["StrategyBucket", "MonitorGroup"]:
        if col in acct.columns:
            acct[col] = acct[col].astype(str).fillna("Unknown")

    save_csv(acct, os.path.join(out_dir, "cleaned", "AccountInformation_clean.csv"))
    return acct


def clean_index_returns(path: str, out_dir: str) -> pd.DataFrame:
    """
    Clean DailyIndexReturns.csv and compute daily percentage returns (wide format).
    Expected input columns: Date, SYM, PX_LAST
    Output: cleaned/DailyIndexReturns_returns.csv
    """
    idx = pd.read_csv(path)
    idx.columns = [c.strip() for c in idx.columns]

    # Parse date
    idx["Date"] = _to_datetime_safe(idx["Date"])

    # Pivot to wide prices and convert to numeric
    prices = (
        idx.pivot(index="Date", columns="SYM", values="PX_LAST")
        .sort_index()
        .apply(pd.to_numeric, errors="coerce")
    )

    # Daily returns by column
    returns = prices.pct_change()

    save_csv(returns.reset_index(), os.path.join(out_dir, "cleaned", "DailyIndexReturns_returns.csv"))
    return returns.reset_index()


def clean_aum(path: str, out_dir: str) -> pd.DataFrame:
    aum = pd.read_csv(path)
    aum["AsOfDate"] = pd.to_datetime(aum["AsOfDate"], errors="coerce")
    aum["Fund"] = aum["Fund"].astype(str).str.strip()
    aum["AUM"] = pd.to_numeric(aum["AUM"], errors="coerce").fillna(0.0)
    out_path = os.path.join(out_dir, "cleaned", "AUM_clean.csv")
    save_csv(aum, out_path)
    return aum
    

# def integrate_internal(positions: pd.DataFrame,
#                        reference: pd.DataFrame,
#                        accounts: pd.DataFrame,
#                        out_dir: str) -> pd.DataFrame:
#     """
#     Integrate positions with reference (security attributes) and accounts (account attributes).
#     - DO NOT merge AUM here; keep benchmark/AUM joins for analysis-time.
#     - Join keys:
#         * positions ↔ reference on ['SEDOL', 'Month'] (monthly attributes broadcast to daily)
#         * positions ↔ accounts  on ['Account']
#     """
#     # Ensure join keys exist
#     if "Month" not in positions.columns and "Date" in positions.columns:
#         positions = positions.copy()
#         positions["Month"] = _month_str_from_date(positions["Date"])
#     if "Month" not in reference.columns and "Date" in reference.columns:
#         reference = reference.copy()
#         reference["Month"] = _month_str_from_date(reference["Date"])

#     # Minimal reference columns
#     ref_cols = [c for c in [
#         "SEDOL", "Month",
#         "GICS_SECTOR_CLEAN", "Side",
#         "AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"
#     ] if c in reference.columns]

#     # Merge positions + reference
#     df = positions.merge(
#         reference[ref_cols].drop_duplicates(),
#         on=["SEDOL", "Month"],
#         how="left"
#     )

#     # Select minimal columns needed from accounts
#     acct_cols = [c for c in [
#         "Account", "MonitorGroup", "StrategyBucket",
#         "StaticNotional", "DrawDownLimPctGMV", "DrawdownLimMultipleOfAnnualizedVol"
#     ] if c in accounts.columns]

#     # Merge accounts
#     df = df.merge(
#         accounts[acct_cols].drop_duplicates(),
#         on="Account",
#         how="left"
#     )

#     save_csv(df, os.path.join(out_dir, "integrated", "TooSharpe_InternalIntegrated.csv"))
#     return df

def audit_merge_na(integrated: pd.DataFrame, out_dir: str) -> None:
    """
    Identify rows where reference-derived fields are NA after merge.
    Save them to outputs/dropped/integrated_missing_reference.csv for audit.
    """
    ref_cols = ["GICS_SECTOR_CLEAN", "Side", "AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"]
    ref_cols = [c for c in ref_cols if c in integrated.columns]
    if not ref_cols:
        print("⚠️ No reference columns found for NA audit.")
        return

    missing_mask = integrated[ref_cols].isna().all(axis=1)
    missing_rows = integrated[missing_mask].copy()
    count = len(missing_rows)
    if count == 0:
        print("✅ No NA rows found after merge with reference.")
        return

    dropped_dir = os.path.join(out_dir, "dropped")
    os.makedirs(dropped_dir, exist_ok=True)
    out_path = os.path.join(dropped_dir, "integrated_missing_reference.csv")
    missing_rows.to_csv(out_path, index=False)

    print(f"⚠️ {count} rows have NA in all reference columns — saved to {out_path}")
    print("Reason: These securities/months had no match in reference data.")


# def integrate_internal(positions: pd.DataFrame,
#                        reference: pd.DataFrame,
#                        accounts: pd.DataFrame,
#                        out_dir: str) -> pd.DataFrame:
#     """
#     Integrate positions with reference (security attributes) and accounts (account attributes).
#     Joins:
#       positions ↔ reference on ['SEDOL', 'Month']  (monthly attributes broadcast to daily)
#       positions ↔ accounts  on ['Account']
#     """

#     df_pos = positions.copy()
#     df_ref = reference.copy()
#     df_acct = accounts.copy()

#     # --- Key normalization (defensive and consistent) ---
#     for df in (df_pos, df_ref, df_acct):
#         if "SEDOL" in df.columns:
#             df["SEDOL"] = df["SEDOL"].astype("string").str.strip()
#         if "Account" in df.columns:
#             df["Account"] = df["Account"].astype("string").str.strip()
#         if "Date" in df.columns:
#             df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str)

#     # Drop rows where join keys are missing on the left table only (positions)
#     df_pos = df_pos.dropna(subset=["SEDOL", "Month", "Account"])

#     # --- Reference subset ---
#     ref_cols = [c for c in [
#         "SEDOL", "Month",
#         "GICS_SECTOR_CLEAN", "Side",
#         "AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"
#     ] if c in df_ref.columns]
#     df_ref = df_ref[ref_cols].drop_duplicates()

#     # --- Accounts subset ---
#     acct_cols = [c for c in [
#         "Account", "MonitorGroup", "StrategyBucket",
#         "StaticNotional", "DrawDownLimPctGMV", "DrawdownLimMultipleOfAnnualizedVol"
#     ] if c in df_acct.columns]
#     df_acct = df_acct[acct_cols].drop_duplicates()

#     # --- Merge ---
#     merged = df_pos.merge(df_ref, on=["SEDOL", "Month"], how="left", validate="m:1")
#     merged = merged.merge(df_acct, on="Account", how="left", validate="m:1")

#     # Persist
#     save_csv(merged, os.path.join(out_dir, "integrated", "TooSharpe_InternalIntegrated.csv"))
#     audit_merge_na(merged, out_dir)

#     # Remove na
#     merged_remove_na = merged.dropna()
    
#     return merged_remove_na


def integrate_internal(positions: pd.DataFrame,
                       reference: pd.DataFrame,
                       accounts: pd.DataFrame,
                       out_dir: str) -> pd.DataFrame:
    """
    Integrate positions with reference (security attributes) and accounts (account attributes).
    Drop rows with completely missing reference attributes (after saving audit report).
    """
    df_pos = positions.copy()
    df_ref = reference.copy()
    df_acct = accounts.copy()

    # Normalize join keys
    for df in (df_pos, df_ref, df_acct):
        if "SEDOL" in df.columns:
            df["SEDOL"] = df["SEDOL"].astype("string").str.strip()
        if "Account" in df.columns:
            df["Account"] = df["Account"].astype("string").str.strip()
        if "Date" in df.columns:
            df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str)

    df_pos = df_pos.dropna(subset=["SEDOL", "Month", "Account"])

    # Reference + Account subsets
    ref_cols = ["SEDOL", "Month", "GICS_SECTOR_CLEAN", "Side", "AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"]
    ref_cols = [c for c in ref_cols if c in df_ref.columns]
    df_ref = df_ref[ref_cols].drop_duplicates()

    acct_cols = ["Account", "MonitorGroup", "StrategyBucket", "StaticNotional",
                 "DrawDownLimPctGMV", "DrawdownLimMultipleOfAnnualizedVol"]
    acct_cols = [c for c in acct_cols if c in df_acct.columns]
    df_acct = df_acct[acct_cols].drop_duplicates()

    # Merge
    merged = df_pos.merge(df_ref, on=["SEDOL", "Month"], how="left", validate="m:1")
    merged = merged.merge(df_acct, on="Account", how="left", validate="m:1")

    # --- Audit & drop NA reference rows ---
    ref_attr = [c for c in ["GICS_SECTOR_CLEAN", "Side", "AverageDailyVolume_20_DAY", "MARKET_CAP", "ETF_C"]
                if c in merged.columns]
    na_mask = merged[ref_attr].isna().all(axis=1)

    if na_mask.any():
        dropped_dir = os.path.join(out_dir, "dropped")
        os.makedirs(dropped_dir, exist_ok=True)
        dropped_path = os.path.join(dropped_dir, "integrated_missing_reference.csv")
        merged[na_mask].to_csv(dropped_path, index=False)
        print(f"🧹 Dropped {na_mask.sum()} rows missing all reference attributes → {dropped_path}")

    merged_clean = merged[~na_mask].copy()

    save_csv(merged_clean, os.path.join(out_dir, "integrated", "TooSharpe_InternalIntegrated.csv"))
    return merged_clean

# def build_manifest(out_dir: str) -> dict:
#     manifest = {
#         "generated_at": datetime.utcnow().isoformat() + "Z",
#         "paths": {
#             "clean_reference": os.path.join(out_dir, "cleaned", "ReferenceData_clean.csv"),
#             "clean_positions": os.path.join(out_dir, "cleaned", "PositionLevelPNLAndExposure_clean.csv"),
#             "clean_accounts": os.path.join(out_dir, "cleaned", "AccountInformation_clean.csv"),
#             "clean_index_returns": os.path.join(out_dir, "cleaned", "DailyIndexReturns_returns.csv"),
#             "clean_aum": os.path.join(out_dir, "cleaned", "AUM_clean.csv"),
#             "integrated_internal": os.path.join(out_dir, "integrated", "TooSharpe_InternalIntegrated.csv"),
#         }
#     }
#     ensure_dir(out_dir)
#     with open(os.path.join(out_dir, "manifest.json"), "w") as f:
#         json.dump(manifest, f, indent=2)
#     return manifest


def build_manifest(out_dir: str) -> dict:
    ensure_dir(out_dir)
    paths = {
        "clean_reference": os.path.join(out_dir, "cleaned", "ReferenceData_clean.csv"),
        "clean_positions": os.path.join(out_dir, "cleaned", "PositionLevelPNLAndExposure_clean.csv"),
        "clean_accounts": os.path.join(out_dir, "cleaned", "AccountInformation_clean.csv"),
        "clean_index_returns": os.path.join(out_dir, "cleaned", "DailyIndexReturns_returns.csv"),
        "clean_aum": os.path.join(out_dir, "cleaned", "AUM_clean.csv"),
        "integrated_internal": os.path.join(out_dir, "integrated", "TooSharpe_InternalIntegrated.csv"),
        # dropped artifacts (conditionally present)
        "dropped_positions_singleton_sedol": os.path.join(out_dir, "dropped", "positions_singleton_sedol.csv"),
    }
    # 只把存在的 dropped 文件写入（更干净）
    paths = {k: v for k, v in paths.items() if (k.startswith("dropped_") and os.path.exists(v)) or not k.startswith("dropped_")}

    manifest = {"generated_at": datetime.utcnow().isoformat() + "Z", "paths": paths}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="TooSharpe Data Handling (clean + internal integrate; no row-level benchmark join)")
    parser.add_argument("--raw-dir", type=str, required=True, help="Directory containing raw CSV files")
    parser.add_argument("--out-dir", type=str, required=True, help="Base output directory")
    parser.add_argument("--ref-file", type=str, default="ReferenceData.csv")
    parser.add_argument("--pos-file", type=str, default="PositionLevelPNLAndExposure.csv")
    parser.add_argument("--acct-file", type=str, default="AccountInformation.csv")
    parser.add_argument("--idx-file", type=str, default="DailyIndexReturns.csv")
    parser.add_argument("--aum-file", type=str, default="AUM.csv")
    args = parser.parse_args()

    raw = args.raw_dir
    out = args.out_dir
    ensure_dir(out)

    # Clean
    ref  = clean_reference_data(os.path.join(raw, args.ref_file), out)
    pos  = clean_positions(os.path.join(raw, args.pos_file), out)
    acct = clean_accounts(os.path.join(raw, args.acct_file), out)
    idx  = clean_index_returns(os.path.join(raw, args.idx_file), out)
    aum  = clean_aum(os.path.join(raw, args.aum_file), out)

    # Integrate internal only (no benchmark join here)
    integrated = integrate_internal(pos, ref, acct, out)

    # Manifest
    manifest = build_manifest(out)

    # Small log
    print("Done. Outputs written.")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()