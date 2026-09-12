"""
Column-mapping config for the DoD FPDS-NG construction contract dataset
(Mendeley DOI: 10.17632/yk4s7pdsvk.1).

Why this exists: FPDS-NG exports have used slightly different header names
across query tools/years (e.g. "Base And All Options Value" vs
"baseAndAllOptionsValue"). Rather than hard-coding one exact header, each
field below lists CANDIDATE header strings. The loader auto-matches the
first candidate found in your actual CSV (case-insensitive, punctuation-
insensitive). If none match, it raises a clear error listing your real
column names so you can add the correct one to the list in 5 seconds.

>>> EDIT THIS FILE after you've loaded the real CSV once and seen its
>>> actual headers via `python run_experiment.py --inspect path/to/file.csv`
"""

COLUMN_CANDIDATES = {
    # --- identifiers ---
    "contract_id": ["PIID", "Contract ID", "Award ID", "procurement_instrument_identifier"],

    # --- dates (used to build schedule targets) ---
    "award_date": ["Date Signed", "Award Date", "signeddate", "effective_date"],
    "start_date": ["Period Of Performance Start Date", "period_of_performance_start_date",
                   "Effective Date", "start_date"],
    "current_completion_date": ["Current Completion Date", "current_completion_date",
                                 "Period Of Performance Current End Date"],
    "ultimate_completion_date": ["Ultimate Completion Date", "ultimate_completion_date",
                                 "Period Of Performance Potential End Date"],

    # --- cost (used to build cost targets) ---
    "base_and_options_value": ["Base And All Options Value", "base_and_all_options_value",
                                "Base and Options Value"],
    "current_total_value": ["Current Total Value Of Award", "dollarsobligated",
                             "Action Obligation", "current_total_value_of_award"],

    # --- categorical / context features ---
    "contracting_agency": ["Contracting Agency ID", "contractingofficeagencyid", "Funding Agency"],
    "military_branch": ["Military Department", "contracting_department_name", "Component"],
    "naics_code": ["NAICS Code", "naics", "principal_naics_code"],
    "psc_code": ["Product Or Service Code", "PSC Code", "productorservicecode"],
    "contract_type": ["Type Of Contract", "contract_pricing", "type_of_contract"],
    "extent_competed": ["Extent Competed", "extentcompeted"],
    "number_of_offers": ["Number Of Offers Received", "number_of_offers_received"],
    "place_state": ["Place Of Performance State Code", "pop_state_code", "State"],
    "fiscal_year": ["Fiscal Year", "fiscal_year", "contractfiscal_year"],
}

# Risk label thresholds (edit if your supervisor wants different cutoffs;
# justify whatever you pick in the methodology chapter).
COST_OVERRUN_THRESHOLD = 0.10     # >10% growth from base value = cost risk flag
SCHEDULE_OVERRUN_THRESHOLD_DAYS = 30  # >30 days beyond current vs ultimate completion date

# Temporal split (train on earlier fiscal years, test on later ones —
# this is the realistic "forecast future projects" setup, not random split).
#
# IMPORTANT — right-censoring: this dataset was extracted mid-2020, so
# projects awarded in 2018-2020 haven't had enough elapsed time to
# accumulate modifications; their cost_growth/schedule_slip values are
# artificially near-zero, not genuinely low-risk. Verified empirically:
# mean transactions-per-project falls from ~22 (FY2009) to ~1.7 (FY2019).
# Excluding 2018-2020 avoids training/testing against censored labels.
MAX_MATURE_FISCAL_YEAR = 2017   # drop any project awarded after this year
TRAIN_YEARS_MAX = 2014          # train: <=2014, test: 2015-2017 (both mature)
