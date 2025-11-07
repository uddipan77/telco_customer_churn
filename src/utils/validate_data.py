# src/utils/validate_data.py
from __future__ import annotations
from typing import Tuple, List
import pandas as pd

def _non_null_count(s: pd.Series) -> int:
    return int(s.isna().sum())

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Lightweight data validation for the Telco Customer Churn dataset.
    Allows blank/non-numeric TotalCharges for brand-new customers (tenure == 0),
    which is a known quirk of this dataset.
    Returns (is_valid, failures_list).
    """
    print("🔍 Starting data validation (lightweight checks, no GE)...")

    failures: List[str] = []

    # ---------- SCHEMA / REQUIRED COLUMNS ----------
    print("   📋 Validating schema and required columns...")
    required_columns = {
        "customerID","gender","Partner","Dependents","PhoneService","InternetService",
        "Contract","tenure","MonthlyCharges","TotalCharges","Churn"
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        failures.append(f"Missing required columns: {missing}")
        print("   ❌ Schema validation failed.")
        return False, failures

    # ---------- BASIC NULL CHECKS ----------
    key_not_null = ["customerID", "tenure", "MonthlyCharges"]
    for col in key_not_null:
        nn = _non_null_count(df[col])
        if nn > 0:
            failures.append(f"Column '{col}' contains {nn} nulls")

    # ---------- VALUE SET CHECKS ----------
    print("   💼 Validating business logic constraints (value sets)...")
    def check_in_set(col: str, allowed: set[str]) -> None:
        bad = sorted(set(df[col].dropna().unique()) - allowed)
        if bad:
            failures.append(f"Unexpected values in '{col}': {bad}; allowed={sorted(allowed)}")

    check_in_set("gender", {"Male", "Female"})
    check_in_set("Partner", {"Yes", "No"})
    check_in_set("Dependents", {"Yes", "No"})
    check_in_set("PhoneService", {"Yes", "No"})
    check_in_set("Contract", {"Month-to-month", "One year", "Two year"})
    check_in_set("InternetService", {"DSL", "Fiber optic", "No"})

    # ---------- NUMERIC RANGE + SPECIAL HANDLING FOR TotalCharges ----------
    print("   📊 Validating numeric ranges and constraints...")
    # Known quirk: some rows have empty/blank TotalCharges for tenure == 0.
    tc_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
    non_numeric_mask = tc_numeric.isna() & df["TotalCharges"].notna()
    # Rows where TotalCharges is non-numeric but tenure == 0 are acceptable
    ok_blank_mask = non_numeric_mask & (df["tenure"] == 0)
    real_issues_mask = non_numeric_mask & (df["tenure"] > 0)
    if real_issues_mask.any():
        failures.append(
            f"'TotalCharges' has {int(real_issues_mask.sum())} non-numeric values where tenure > 0"
        )
    elif non_numeric_mask.any():
        # Only blanks with tenure==0 → warn but don't fail
        print(f"   ⚠️  {int(non_numeric_mask.sum())} non-numeric 'TotalCharges' for tenure==0; allowed")

    # Tenure, MonthlyCharges, TotalCharges non-negative
    if (df["tenure"] < 0).any():
        failures.append("'tenure' contains negative values")
    if (df["MonthlyCharges"] < 0).any():
        failures.append("'MonthlyCharges' contains negative values")
    if (tc_numeric.dropna() < 0).any():
        failures.append("'TotalCharges' contains negative values")

    # Reasonable business bounds
    if (df["tenure"] > 120).any():
        failures.append("'tenure' exceeds 120 months (10 years) in some rows")
    if (df["MonthlyCharges"] > 200).any():
        failures.append("'MonthlyCharges' exceeds 200 in some rows")

    # Ensure critical numeric columns not null
    if df["tenure"].isna().any():
        failures.append(f"'tenure' has {int(df['tenure'].isna().sum())} nulls")
    if df["MonthlyCharges"].isna().any():
        failures.append(f"'MonthlyCharges' has {int(df['MonthlyCharges'].isna().sum())} nulls")

    # ---------- CONSISTENCY: TotalCharges >= MonthlyCharges for ≥95% ----------
    print("   🔗 Validating data consistency...")
    comparable = (~tc_numeric.isna()) & (~df["MonthlyCharges"].isna())
    if comparable.any():
        violations = (tc_numeric[comparable] < df.loc[comparable, "MonthlyCharges"]).sum()
        total = int(comparable.sum())
        frac_ok = 1.0 - (violations / total)
        if frac_ok < 0.95:
            failures.append(
                f"'TotalCharges >= MonthlyCharges' holds for {frac_ok:.3%} of rows; "
                f"expected ≥ 95% (violations={int(violations)}/{total})"
            )
    else:
        # If we have no comparable rows, that itself is suspicious
        failures.append("Not enough non-null rows to evaluate 'TotalCharges >= MonthlyCharges'")

    # Target sanity
    extras = sorted(set(df["Churn"].dropna().unique()) - {"Yes", "No"})
    if extras:
        failures.append(f"Unexpected values in 'Churn': {extras} (expected 'Yes'/'No')")

    # customerID uniqueness
    dupes = int(df["customerID"].duplicated().sum())
    if dupes > 0:
        failures.append(f"'customerID' has {dupes} duplicate values")

    if not failures:
        print("✅ Data validation PASSED")
        return True, []
    else:
        print("❌ Data validation FAILED")
        for msg in failures:
            print(f"   - {msg}")
        return False, failures
