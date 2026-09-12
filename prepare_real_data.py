"""
The raw FPDSData.csv is TRANSACTION-level: one row per base award AND one
row per subsequent modification to that same contract (360,029 rows total,
of which exactly 132,662 have Modification Number == '0', i.e. base awards
— matching the dataset's documented project count exactly).

To measure real cost growth / schedule slip we must link each base award to
its own later modifications and compare "at signing" vs "latest known"
values. There is no single clean primary key across agencies/offices (PIID
alone repeats across different offices), so we use a composite key:

    Contracting Agency ID + Contracting Office ID + PIID

Sanity-checked on the real file: 88,201 unique keys; of the 51,095 keys with
more than one row, 80% span <= 3 years between first and last transaction
(consistent with being a genuine modification chain on one contract, not a
coincidental key collision) and under 0.1% span more than 10 years. This is
a known real-data linkage imperfection — document it as a limitation in your
methodology chapter rather than presenting the linkage as perfect.

Within each key-group:
    base record   = row with the EARLIEST Date Signed  (the original award)
    latest record = row with the LATEST   Date Signed  (most recent state)

    cost_growth_ratio   = (latest total value - base total value) / base total value
    schedule_slip_days  = latest Est. Ultimate Completion Date
                          - base Completion Date
"""
import pandas as pd
import numpy as np

RAW_COLUMNS = {
    "agency": "Contracting Agency ID",
    "office": "Contracting Office ID",
    "piid": "PIID",
    "date_signed": "Date Signed",
    "effective_date": "Effective Date",
    "completion_date": "Completion Date",
    "ultimate_completion_date": "Est. Ultimate Completion Date",
    "total_value": "Base and All Options Value (Total Contract Value)",
    "fiscal_year": "Fiscal Year",
    "military_branch": "Major Command Name",
    "naics_code": "NAICS Code",
    "psc_code": "Product or Service Code",
    "contract_type": "Type of Contract",
    "extent_competed": "Extent Competed",
    "number_of_offers": "Number of Offers Received",
    "place_state": "Principal Place of Performance State Code",
}


def prepare(raw_path: str, out_path: str):
    print(f"Loading raw transaction file {raw_path} ...")
    df = pd.read_csv(raw_path, low_memory=False, usecols=list(RAW_COLUMNS.values()))
    df = df.rename(columns={v: k for k, v in RAW_COLUMNS.items()})
    print(f"  {len(df):,} raw transaction rows")

    df["date_signed"] = pd.to_datetime(df["date_signed"], errors="coerce")
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["ultimate_completion_date"] = pd.to_datetime(df["ultimate_completion_date"], errors="coerce")
    df["total_value"] = pd.to_numeric(df["total_value"], errors="coerce")

    df["key"] = (df["agency"].astype(str) + "_" + df["office"].astype(str) + "_" + df["piid"].astype(str))
    df = df.dropna(subset=["date_signed", "total_value"])
    df = df.sort_values(["key", "date_signed"])

    print("Aggregating transactions into one row per project ...")
    first = df.groupby("key", as_index=False).first()
    last = df.groupby("key", as_index=False).last()

    proj = first[[
        "key", "effective_date", "completion_date", "fiscal_year",
        "military_branch", "naics_code", "psc_code", "contract_type",
        "extent_competed", "number_of_offers", "place_state",
    ]].copy()
    proj = proj.rename(columns={"total_value": "base_value"})
    proj["base_value"] = first["total_value"].values
    proj["final_value"] = last["total_value"].values
    proj["ultimate_completion_date"] = last["ultimate_completion_date"].values
    proj["n_transactions"] = df.groupby("key").size().values

    # --- targets ---
    proj["cost_growth_ratio"] = (proj["final_value"] - proj["base_value"]) / proj["base_value"].replace(0, np.nan)
    proj["schedule_slip_days"] = (proj["ultimate_completion_date"] - proj["completion_date"]).dt.days
    proj["planned_duration_days"] = (proj["completion_date"] - proj["effective_date"]).dt.days

    # drop nonsensical / corrupted rows (negative base value, absurd durations)
    proj = proj[proj["base_value"] > 0]
    proj = proj[(proj["planned_duration_days"] > 0) & (proj["planned_duration_days"] < 3650)]

    keep_cols = ["fiscal_year", "base_value", "cost_growth_ratio", "schedule_slip_days",
                 "planned_duration_days", "number_of_offers", "n_transactions",
                 "military_branch", "naics_code", "psc_code", "contract_type",
                 "extent_competed", "place_state"]
    proj = proj[keep_cols]
    proj.to_csv(out_path, index=False)
    print(f"Wrote {len(proj):,} project-level rows -> {out_path}")
    return proj


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="projects_clean.csv")
    args = p.parse_args()
    prepare(args.raw, args.out)
