"""
Data pipeline: raw FPDS-NG CSV  ->  clean feature matrix + 3 targets
(cost growth ratio, schedule slip in days, binary risk flag).

Usage:
    python run_experiment.py --data /path/to/dod_construction.csv
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import COLUMN_CANDIDATES, COST_OVERRUN_THRESHOLD, \
    SCHEDULE_OVERRUN_THRESHOLD_DAYS, TRAIN_YEARS_MAX, MAX_MATURE_FISCAL_YEAR


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_columns(df: pd.DataFrame) -> dict:
    """Map our canonical field names -> actual column names found in df."""
    normalized_actual = {_normalize(c): c for c in df.columns}
    resolved, missing = {}, []
    for field, candidates in COLUMN_CANDIDATES.items():
        found = None
        for cand in candidates:
            key = _normalize(cand)
            if key in normalized_actual:
                found = normalized_actual[key]
                break
        if found:
            resolved[field] = found
        else:
            missing.append(field)
    if missing:
        print("\n[WARN] Could not auto-match these fields:", missing)
        print("Actual columns in your file are:\n", list(df.columns))
        print("-> Add the correct header text to config.py's COLUMN_CANDIDATES "
              "for each missing field, then re-run.\n")
    return resolved


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {path}")
    return df


def engineer_features(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # --- dates ---
    for f in ["award_date", "start_date", "current_completion_date", "ultimate_completion_date"]:
        if f in cols:
            out[f] = pd.to_datetime(df[cols[f]], errors="coerce")

    # --- cost target: growth ratio vs base value ---
    if "base_and_options_value" in cols and "current_total_value" in cols:
        base = pd.to_numeric(df[cols["base_and_options_value"]], errors="coerce")
        current = pd.to_numeric(df[cols["current_total_value"]], errors="coerce")
        out["base_value"] = base
        out["cost_growth_ratio"] = (current - base) / base.replace(0, np.nan)

    # --- schedule target: slip in days ---
    if "current_completion_date" in cols and "ultimate_completion_date" in cols:
        out["schedule_slip_days"] = (
            out["ultimate_completion_date"] - out["current_completion_date"]
        ).dt.days

    # --- project duration (a feature, not a target) ---
    if "start_date" in out.columns and "current_completion_date" in out.columns:
        out["planned_duration_days"] = (
            out["current_completion_date"] - out["start_date"]
        ).dt.days

    # --- fiscal year (for temporal split) ---
    if "fiscal_year" in cols:
        out["fiscal_year"] = pd.to_numeric(df[cols["fiscal_year"]], errors="coerce")
    elif "award_date" in out.columns:
        out["fiscal_year"] = out["award_date"].dt.year

    # --- categorical context features (kept as raw strings; encoded later) ---
    for f in ["contracting_agency", "military_branch", "naics_code", "psc_code",
              "contract_type", "extent_competed", "place_state"]:
        if f in cols:
            out[f] = df[cols[f]].astype(str)

    if "number_of_offers" in cols:
        out["number_of_offers"] = pd.to_numeric(df[cols["number_of_offers"]], errors="coerce")

    # --- risk label: cost OR schedule overrun beyond threshold ---
    cost_flag = out.get("cost_growth_ratio", pd.Series(0, index=out.index)) > COST_OVERRUN_THRESHOLD
    sched_flag = out.get("schedule_slip_days", pd.Series(0, index=out.index)) > SCHEDULE_OVERRUN_THRESHOLD_DAYS
    out["schedule_overrun_label"] = sched_flag.astype(int)
    out["risk_label"] = (cost_flag | sched_flag).astype(int)

    return out


def add_risk_label(feat: pd.DataFrame) -> pd.DataFrame:
    """For already-aggregated project-level data (see prepare_real_data.py)
    that doesn't yet have risk_label / schedule_overrun_label columns.

    schedule_overrun_label is now a BINARY classification target instead of
    raw schedule_slip_days regression — the raw regression version scored
    strongly negative R2 in testing (heavy-tailed, censored real data), so
    schedule is reframed as "will this project slip beyond the threshold?"
    which is both more learnable and arguably more decision-relevant anyway.
    """
    cost_flag = feat.get("cost_growth_ratio", pd.Series(0, index=feat.index)) > COST_OVERRUN_THRESHOLD
    sched_flag = feat.get("schedule_slip_days", pd.Series(0, index=feat.index)) > SCHEDULE_OVERRUN_THRESHOLD_DAYS
    feat = feat.copy()
    feat["schedule_overrun_label"] = sched_flag.astype(int)
    feat["risk_label"] = (cost_flag | sched_flag).astype(int)
    return feat


def clean_and_split(feat: pd.DataFrame):
    """Drop rows missing any target, encode categoricals, scale numerics,
    do a TEMPORAL split (train on earlier years, test on later years)."""
    targets = ["cost_growth_ratio", "schedule_overrun_label", "risk_label"]
    feat = feat.dropna(subset=[t for t in targets if t in feat.columns])

    if "fiscal_year" in feat.columns:
        before = len(feat)
        feat = feat[feat["fiscal_year"] <= MAX_MATURE_FISCAL_YEAR]
        print(f"Dropped {before - len(feat):,} right-censored projects awarded after "
              f"FY{MAX_MATURE_FISCAL_YEAR} (not enough elapsed time to show true overruns)")

    # clip extreme outliers (winsorize at 1st/99th pct) — construction cost
    # ratios have a long tail; this matters for MSE-based loss stability
    if "cost_growth_ratio" in feat.columns:
        lo, hi = feat["cost_growth_ratio"].quantile([0.01, 0.99])
        feat["cost_growth_ratio"] = feat["cost_growth_ratio"].clip(lo, hi)

    cat_cols = [c for c in ["contracting_agency", "military_branch", "naics_code",
                             "psc_code", "contract_type", "extent_competed", "place_state"]
                if c in feat.columns]
    num_cols = [c for c in ["base_value", "planned_duration_days", "number_of_offers"]
                if c in feat.columns]

    # frequency-encode high-cardinality categoricals (NAICS/PSC have hundreds
    # of codes; one-hot would blow up dimensionality)
    for c in cat_cols:
        freq = feat[c].value_counts(normalize=True)
        feat[c + "_freq"] = feat[c].map(freq)
    encoded_cat_cols = [c + "_freq" for c in cat_cols]

    feature_cols = num_cols + encoded_cat_cols
    feat[feature_cols] = feat[feature_cols].fillna(0)

    if "fiscal_year" not in feat.columns:
        raise ValueError("fiscal_year could not be derived — cannot do temporal split.")

    train_mask = feat["fiscal_year"] <= TRAIN_YEARS_MAX
    test_mask = ~train_mask

    X_train_raw = feat.loc[train_mask, feature_cols].values
    X_test_raw = feat.loc[test_mask, feature_cols].values

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = {t: feat.loc[train_mask, t].values for t in targets if t in feat.columns}
    y_test = {t: feat.loc[test_mask, t].values for t in targets if t in feat.columns}

    print(f"Train: {X_train.shape[0]:,} projects (<= FY{TRAIN_YEARS_MAX}) | "
          f"Test: {X_test.shape[0]:,} projects (> FY{TRAIN_YEARS_MAX}) | "
          f"{X_train.shape[1]} features")

    return X_train, X_test, y_train, y_test, feature_cols
