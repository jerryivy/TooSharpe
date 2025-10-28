import os
import pandas as pd


BASE_NULL_TOKENS = {"", "(null)", "null", "none", "na", "nan"}


def _to_datetime_safe(series, dayfirst=False):
    """Safely convert a Series to datetime (coerce errors)."""
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)

def _month_str_from_date(series):
    """Convert datetime Series to 'YYYY-MM' strings."""
    return _to_datetime_safe(series).dt.to_period("M").astype(str)

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV, creating parent dirs."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def summarize_nulls(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Null% by column (for quick sanity checks)."""
    s = df.isna().mean().sort_values(ascending=False) * 100
    return pd.DataFrame({"column": s.index, f"null_pct_{tag}": s.values.round(2)})

def ffill_series_to_daily(df: pd.DataFrame, date_col: str, value_col: str,
                          start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Reindex a dated series to daily frequency and forward-fill."""
    s = (df.sort_values(date_col).set_index(date_col)[[value_col]])
    idx = pd.date_range(start_date, end_date, freq="D")
    out = s.reindex(idx, method="ffill").rename_axis("Date").reset_index()
    out = out.rename(columns={value_col: value_col})
    return out

def normalize_str_nulls(series: pd.Series, to_na: bool = True) -> pd.Series:
    """Trim spaces and map common null-like tokens to pandas NA (case-insensitive)."""
    s = series.astype("string").str.strip()
    mask = s.str.lower().isin(BASE_NULL_TOKENS)
    return s.mask(mask, pd.NA) if to_na else s

def clean_categorical(series: pd.Series, default: str = "Unknown") -> pd.Series:
    """Normalize string nulls then fill with a default label."""
    return normalize_str_nulls(series, to_na=True).fillna(default)

# def impute_by_group_unique(
#     df: pd.DataFrame, group_keys: list[str], target_cols: list[str], unknown: str = "Unknown"
# ) -> pd.DataFrame:
#     """
#     For each target categorical column:
#       - if within a group (defined by group_keys) the non-null values are UNIQUE (exactly one),
#         fill missing with that unique value;
#       - otherwise fill missing with `unknown`.
#     Assumes target columns are string-like and already normalized to NA for null-likes.
#     """
#     out = df.copy()
#     for col in target_cols:
#         if col not in out.columns:
#             continue
#         # ensure string dtype and normalize nulls
#         out[col] = normalize_str_nulls(out[col], to_na=True)

#         # groups with exactly one non-null unique value
#         non_null = out.dropna(subset=[col])
#         uniq_counts = non_null.groupby(group_keys)[col].nunique(dropna=True)
#         unique_groups = uniq_counts[uniq_counts == 1].index

#         # mapping: group_keys -> the unique value
#         unique_map = non_null.groupby(group_keys)[col].first()

#         # build a Series aligned to df rows with the unique value (or NA if group not unique)
#         mapped = (
#             out[group_keys]
#             .merge(unique_map.reset_index(), on=group_keys, how="left", suffixes=("", "_unique"))
#             [col]  # this is the merged unique value column
#         )

#         # fill: if NA and group unique -> mapped value; else -> `unknown`
#         need_fill = out[col].isna()
#         out.loc[need_fill, col] = mapped[need_fill]
#         still_na = out[col].isna()
#         out.loc[still_na, col] = unknown
#     return out

def impute_by_group_unique(
    df: pd.DataFrame, group_keys: list[str], target_cols: list[str], unknown: str = "Unknown"
) -> pd.DataFrame:
    """
    For each target categorical column:
      - If within a group (defined by group_keys) the non-null values are UNIQUE (exactly one),
        fill missing with that unique value;
      - Otherwise fill missing with `unknown`.
    Index-safe: mapped series is aligned to df.index.
    """
    out = df.copy()

    for col in target_cols:
        if col not in out.columns:
            continue

        # groups with exactly one non-null unique value
        non_null = out.dropna(subset=[col])

        if non_null.empty:
            # fill NA as unknown
            out[col] = out[col].fillna(unknown)
            continue

        # mapping: group_keys -> the unique value
        uniq_map = (
            non_null.groupby(group_keys, as_index=False)[col]
            .agg(lambda s: s.iloc[0] if s.nunique(dropna=True) == 1 else pd.NA)
        )

        # build a Series aligned to df rows with the unique value (or NA if group not unique)
        mapped = out[group_keys].merge(uniq_map, on=group_keys, how="left")[col]
        mapped.index = out.index  

        # fill: if NA and group unique -> mapped value; else -> `unknown`
        out[col] = out[col].fillna(mapped)
        out[col] = out[col].fillna(unknown)

    return out