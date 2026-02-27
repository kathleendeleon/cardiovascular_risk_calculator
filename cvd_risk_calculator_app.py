import io
import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# pyreadr reads R .rda / .rds files
import pyreadr


# ----------------------------
# Config / constants
# ----------------------------
CRAN_PREVENTR_TARBALL = "https://cran.r-project.org/src/contrib/preventr_1.1.1.tar.gz"
# If CRAN updates, you can swap to:
# https://cran.r-project.org/src/contrib/preventr_latest.tar.gz  (not always available)
# or read package index and resolve, but keeping it simple.

# PREVENT uses cholesterol unit handling; we’ll support mg/dL and mmol/L
MGDL_TO_MMOL_CHOL = 0.02586  # mmol/L = mg/dL * 0.02586


# ----------------------------
# Utility: column mapping
# ----------------------------
DEFAULT_COL_MAP = {
    # PREVENT required
    "age": ["age", "Age", "years", "years_old"],
    "sex": ["sex", "Sex", "gender", "Gender"],
    "sbp": ["sbp", "SBP", "systolic_bp", "systolic", "systolic.bp"],
    "bp_tx": ["bp_tx", "bp_treatment", "on_bp_meds", "treated_bp", "bp_med", "bpmeds"],
    "total_c": ["total_c", "total_chol", "tc", "chol_total", "total.chol"],
    "hdl_c": ["hdl_c", "hdl", "hdl_chol", "total_hdl", "total.hdl"],
    "statin": ["statin", "on_statin", "statins"],
    "dm": ["dm", "diabetes", "t2d", "has_diabetes"],
    "smoking": ["smoking", "smoker", "current_smoker"],

    # PREVENT required (kidney/metabolic)
    "egfr": ["egfr", "eGFR"],
    "bmi": ["bmi", "BMI"],

    # PREVENT optional add-ons
    "hba1c": ["hba1c", "HbA1c", "a1c"],
    "uacr": ["uacr", "UACR", "acr", "albumin_creatinine_ratio"],
    "zip": ["zip", "zipcode", "postal_code"],

    # SCORE2 expects mmol/L for chol; we’ll convert if needed
    "diabetes_score2": ["dm", "diabetes", "t2d", "has_diabetes"],  # reuse dm
    "smoker_score2": ["smoking", "smoker", "current_smoker"],

    # Framingham points table in the CCS PDF uses mmol/L for chol/HDL
}


def resolve_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_sex(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["f", "female", "woman", "w"]:
        return "female"
    if s in ["m", "male", "man"]:
        return "male"
    return None


def to_binary(x) -> Optional[int]:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer, float, np.floating)):
        # accept 0/1 or truthy
        return int(float(x) >= 0.5)
    s = str(x).strip().lower()
    if s in ["1", "true", "t", "yes", "y"]:
        return 1
    if s in ["0", "false", "f", "no", "n"]:
        return 0
    return None


# ----------------------------
# PREVENT: load coefficients from preventr sysdata.rda
# ----------------------------
@dataclass
class PreventModels:
    # Each is a pandas DataFrame with columns like "female_total_cvd", etc.
    base_10yr: pd.DataFrame
    uacr_10yr: pd.DataFrame
    hba1c_10yr: pd.DataFrame
    sdi_10yr: pd.DataFrame
    full_10yr: pd.DataFrame

    base_30yr: pd.DataFrame
    uacr_30yr: pd.DataFrame
    hba1c_30yr: pd.DataFrame
    sdi_30yr: pd.DataFrame
    full_30yr: pd.DataFrame


@st.cache_resource(show_spinner=False)
def load_prevent_models_from_cran() -> PreventModels:
    """
    Downloads preventr tarball, extracts R/sysdata.rda, reads the coefficient tables via pyreadr.
    The R source refers to objects like base_10yr, base_30yr, etc. (stored in sysdata.rda).
    """
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "preventr.tar.gz")
        rda_path = os.path.join(td, "sysdata.rda")

        r = requests.get(CRAN_PREVENTR_TARBALL, timeout=60)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            f.write(r.content)

        with tarfile.open(tar_path, "r:gz") as tar:
            # find the sysdata in the tarball
            members = tar.getmembers()
            sys_members = [m for m in members if m.name.endswith("R/sysdata.rda")]
            if not sys_members:
                raise RuntimeError("Could not find R/sysdata.rda in preventr tarball.")
            tar.extract(sys_members[0], path=td)
            extracted = os.path.join(td, sys_members[0].name)
            os.rename(extracted, rda_path)

        data = pyreadr.read_r(rda_path)  # dict-like of {name: object}

        # Expect these names (as used by preventr::run_models())
        required = [
            "base_10yr", "uacr_10yr", "hba1c_10yr", "sdi_10yr", "full_10yr",
            "base_30yr", "uacr_30yr", "hba1c_30yr", "sdi_30yr", "full_30yr",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(f"Missing objects in sysdata.rda: {missing}. preventr package may have changed.")

        def as_df(obj):
            # pyreadr usually returns pandas.DataFrame already
            if isinstance(obj, pd.DataFrame):
                return obj
            return pd.DataFrame(obj)

        return PreventModels(
            base_10yr=as_df(data["base_10yr"]),
            uacr_10yr=as_df(data["uacr_10yr"]),
            hba1c_10yr=as_df(data["hba1c_10yr"]),
            sdi_10yr=as_df(data["sdi_10yr"]),
            full_10yr=as_df(data["full_10yr"]),
            base_30yr=as_df(data["base_30yr"]),
            uacr_30yr=as_df(data["uacr_30yr"]),
            hba1c_30yr=as_df(data["hba1c_30yr"]),
            sdi_30yr=as_df(data["sdi_30yr"]),
            full_30yr=as_df(data["full_30yr"]),
        )


def convert_chol_to_mmol(total_c, hdl_c, units: str) -> Tuple[float, float]:
    if units == "mg/dL":
        return float(total_c) * MGDL_TO_MMOL_CHOL, float(hdl_c) * MGDL_TO_MMOL_CHOL
    return float(total_c), float(hdl_c)


def prevent_prep_terms(row: dict, chol_units: str) -> dict:
    """
    Mirrors preventr::prep_terms() transformations (centered and scaled predictors, splines, interactions).
    See preventr source. :contentReference[oaicite:3]{index=3}
    """
    age = float(row["age"])
    sbp = float(row["sbp"])
    total_c = float(row["total_c"])
    hdl_c = float(row["hdl_c"])
    bmi = float(row["bmi"])
    egfr = float(row["egfr"])

    dm = int(row["dm"])
    smoking = int(row["smoking"])
    bp_tx = int(row["bp_tx"])
    statin = int(row["statin"])

    total_c_mmol, hdl_c_mmol = convert_chol_to_mmol(total_c, hdl_c, chol_units)
    non_hdl = total_c_mmol - hdl_c_mmol

    pred = {}

    # Centering/scaling as in preventr
    pred["age"] = (age - 55) / 10
    pred["age_squared"] = pred["age"] ** 2

    pred["non_hdl_c"] = non_hdl - 3.5
    pred["hdl_c"] = (hdl_c_mmol - 1.3) / 0.3

    pred["sbp_lt_110"] = (min(sbp, 110) - 110) / 20
    pred["sbp_gte_110"] = (max(sbp, 110) - 130) / 20

    pred["dm"] = dm
    pred["smoking"] = smoking

    pred["bmi_lt_30"] = (min(bmi, 30) - 25) / 5
    pred["bmi_gte_30"] = (max(bmi, 30) - 30) / 5

    pred["egfr_lt_60"] = (min(egfr, 60) - 60) / -15
    pred["egfr_gte_60"] = (max(egfr, 60) - 90) / -15

    pred["bp_tx"] = bp_tx
    pred["statin"] = statin

    # interactions
    pred["bp_tx_sbp_gte_110"] = pred["bp_tx"] * pred["sbp_gte_110"]
    pred["statin_non_hdl_c"] = pred["statin"] * pred["non_hdl_c"]
    pred["age_non_hdl_c"] = pred["age"] * pred["non_hdl_c"]
    pred["age_hdl_c"] = pred["age"] * pred["hdl_c"]
    pred["age_sbp_gte_110"] = pred["age"] * pred["sbp_gte_110"]
    pred["age_dm"] = pred["age"] * pred["dm"]
    pred["age_smoking"] = pred["age"] * pred["smoking"]
    pred["age_bmi_gte_30"] = pred["age"] * pred["bmi_gte_30"]
    pred["age_egfr_lt_60"] = pred["age"] * pred["egfr_lt_60"]

    # Optional add-on terms (present if supplied)
    hba1c = row.get("hba1c", None)
    uacr = row.get("uacr", None)
    sdi = row.get("sdi", None)  # not used unless you implement SDI lookup from ZIP

    # SDI terms: you can add a ZIP->SDI lookup later; for now allow user to pass sdi directly
    if sdi is None or (isinstance(sdi, float) and np.isnan(sdi)):
        pred["sdi_4_to_6"] = 0
        pred["sdi_7_to_10"] = 0
        pred["missing_sdi"] = 1
    else:
        sdi_val = float(sdi)
        pred["sdi_4_to_6"] = int(4 <= sdi_val <= 6)
        pred["sdi_7_to_10"] = int(7 <= sdi_val <= 10)
        pred["missing_sdi"] = 0

    if uacr is None or (isinstance(uacr, float) and np.isnan(uacr)):
        pred["ln_uacr"] = 0.0
        pred["missing_uacr"] = 1
    else:
        pred["ln_uacr"] = float(np.log(float(uacr)))
        pred["missing_uacr"] = 0

    if hba1c is None or (isinstance(hba1c, float) and np.isnan(hba1c)):
        pred["hba1c_dm"] = 0.0
        pred["hba1c_no_dm"] = 0.0
        pred["missing_hba1c"] = 1
    else:
        h = float(hba1c)
        pred["hba1c_dm"] = (h - 5.3) if dm == 1 else 0.0
        pred["hba1c_no_dm"] = (h - 5.3) if dm == 0 else 0.0
        pred["missing_hba1c"] = 0

    pred["constant"] = 1.0
    return pred


def prevent_run_model(coef_df: pd.DataFrame, sex: str, horizon: str, terms: dict) -> Dict[str, float]:
    """
    Mirrors preventr::run_models(): logistic risk = exp(lp)/(1+exp(lp)).
    coef_df columns like 'female_total_cvd', 'female_ascvd', etc; and rows keyed by term names.
    """
    # For 10yr models, drop age_squared term
    use_terms = dict(terms)
    if horizon == "10yr":
        use_terms.pop("age_squared", None)

    # coef_df is shaped (terms x outcomes), but the term names are rownames in R.
    # In sysdata.rda, they come in as a DataFrame with an index column or index.
    # We'll try to use index first; if not present, look for a column named 'term'.
    if coef_df.index.name is None and "term" in coef_df.columns:
        coef_df = coef_df.set_index("term")

    # pick relevant outcome columns
    prefix = f"{sex}_"
    outcome_cols = [c for c in coef_df.columns if c.startswith(prefix)]
    out = {}
    for oc in outcome_cols:
        outcome = oc.replace(prefix, "")
        # Align coefficients with terms
        lp = 0.0
        for tname, tval in use_terms.items():
            if tname in coef_df.index:
                beta = float(coef_df.loc[tname, oc])
                lp += beta * float(tval)
        # logistic
        p = float(np.exp(lp) / (1.0 + np.exp(lp)))
        out[outcome] = p
    return out


def prevent_predict(row: dict, chol_units: str, horizon: str, model_variant: str) -> Dict[str, float]:
    models = load_prevent_models_from_cran()
    sex = row["sex"]

    # choose coefficient table
    table = getattr(models, f"{model_variant}_{horizon}")
    terms = prevent_prep_terms(row, chol_units)

    return prevent_run_model(table, sex=sex, horizon=horizon.replace("yr", "yr"), terms=terms)


# ----------------------------
# SCORE2 (from RiskScorescvd open R code)
# ----------------------------
def score2_risk(region: str, age: float, sex: str, smoker: int, sbp: float, diabetes: int,
               total_chol_mmol: float, hdl_mmol: float) -> float:
    """
    Translated from RiskScorescvd::SCORE2(). :contentReference[oaicite:4]{index=4}
    Returns percent risk (0-100).
    """
    region = region.title()
    if region not in ["Low", "Moderate", "High", "Very High"]:
        raise ValueError("SCORE2 region must be one of: Low, Moderate, High, Very High")

    is_old = age >= 70
    gender = sex  # "male"/"female"

    # region scaling params
    if not is_old:
        if region == "Low" and gender == "male":
            scale1, scale2 = -0.5699, 0.7476
        elif region == "Low" and gender == "female":
            scale1, scale2 = -0.7380, 0.7019
        elif region == "Moderate" and gender == "male":
            scale1, scale2 = -0.1565, 0.8009
        elif region == "Moderate" and gender == "female":
            scale1, scale2 = -0.3143, 0.7701
        elif region == "High" and gender == "male":
            scale1, scale2 = 0.3207, 0.9360
        elif region == "High" and gender == "female":
            scale1, scale2 = 0.5710, 0.9369
        elif region == "Very High" and gender == "male":
            scale1, scale2 = 0.5836, 0.8294
        elif region == "Very High" and gender == "female":
            scale1, scale2 = 0.9412, 0.8329
        else:
            raise ValueError("Invalid SCORE2 combination.")
    else:
        if region == "Low" and gender == "male":
            scale1, scale2 = -0.34, 1.19
        elif region == "Low" and gender == "female":
            scale1, scale2 = -0.52, 1.01
        elif region == "Moderate" and gender == "male":
            scale1, scale2 = 0.01, 1.25
        elif region == "Moderate" and gender == "female":
            scale1, scale2 = -0.1, 1.1
        elif region == "High" and gender == "male":
            scale1, scale2 = 0.08, 1.15
        elif region == "High" and gender == "female":
            scale1, scale2 = 0.38, 1.09
        elif region == "Very High" and gender == "male":
            scale1, scale2 = 0.05, 0.7
        elif region == "Very High" and gender == "female":
            scale1, scale2 = 0.38, 0.69
        else:
            raise ValueError("Invalid SCORE2 combination.")

    if (gender == "male") and (not is_old):
        xx = (
            0.3742 * (age - 60) / 5
            + 0.6012 * smoker
            + 0.2777 * (sbp - 120) / 20
            + 0.6457 * diabetes
            + 0.1458 * (total_chol_mmol - 6) / 1
            + (-0.2698) * (hdl_mmol - 1.3) / 0.5
            + (-0.0755) * (age - 60) / 5 * smoker
            + (-0.0255) * (age - 60) / 5 * (sbp - 120) / 20
            + (-0.0281) * (age - 60) / 5 * (total_chol_mmol - 6) / 1
            + 0.0426 * (age - 60) / 5 * (hdl_mmol - 1.3) / 0.5
            + (-0.0983) * (age - 60) / 5 * diabetes
        )
        xx2 = 1 - (0.9605 ** np.exp(xx))
        xx3 = 1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - xx2))))
        return float(np.round(xx3 * 100, 1))

    if (gender == "female") and (not is_old):
        xx = (
            0.4648 * (age - 60) / 5
            + 0.7744 * smoker
            + 0.3131 * (sbp - 120) / 20
            + 0.8096 * diabetes
            + 0.1002 * (total_chol_mmol - 6) / 1
            + (-0.2606) * (hdl_mmol - 1.3) / 0.5
            + (-0.1088) * (age - 60) / 5 * smoker
            + (-0.0277) * (age - 60) / 5 * (sbp - 120) / 20
            + (-0.0226) * (age - 60) / 5 * (total_chol_mmol - 6) / 1
            + 0.0613 * (age - 60) / 5 * (hdl_mmol - 1.3) / 0.5
            + (-0.1272) * (age - 60) / 5 * diabetes
        )
        xx2 = 1 - (0.9776 ** np.exp(xx))
        xx3 = 1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - xx2))))
        return float(np.round(xx3 * 100, 1))

    if (gender == "male") and is_old:
        xx = (
            0.0634 * (age - 73)
            + 0.4245 * diabetes
            + 0.3524 * smoker
            + 0.0094 * (sbp - 150)
            + 0.0850 * (total_chol_mmol - 6)
            + (-0.3564) * (hdl_mmol - 1.4)
            + (-0.0174) * (age - 73) * diabetes
            + (-0.0247) * (age - 73) * smoker
            + (-0.0005) * (age - 73) * (sbp - 150)
            + 0.0073 * (age - 73) * (total_chol_mmol - 6)
            + 0.0091 * (age - 73) * (hdl_mmol - 1.4)
        )
        xx2 = 1 - (0.7576 ** np.exp(xx - 0.0929))
        xx3 = 1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - xx2))))
        return float(np.round(xx3 * 100, 1))

    if (gender == "female") and is_old:
        xx = (
            0.0789 * (age - 73)
            + 0.6010 * diabetes
            + 0.4921 * smoker
            + 0.0102 * (sbp - 150)
            + 0.0605 * (total_chol_mmol - 6)
            + (-0.3040) * (hdl_mmol - 1.4)
            + (-0.0107) * (age - 73) * diabetes
            + (-0.0255) * (age - 73) * smoker
            + (-0.0004) * (age - 73) * (sbp - 150)
            + (-0.0009) * (age - 73) * (total_chol_mmol - 6)
            + 0.0154 * (age - 73) * (hdl_mmol - 1.4)
        )
        xx2 = 1 - (0.8082 ** np.exp(xx - 0.229))
        xx3 = 1 - np.exp(-np.exp(scale1 + scale2 * np.log(-np.log(1 - xx2))))
        return float(np.round(xx3 * 100, 1))

    raise ValueError("Unhandled SCORE2 case.")


# ----------------------------
# Framingham 2008 (points-based; mmol/L)
# ----------------------------
def framingham_points_2008(sex: str, age: float, total_chol_mmol: float, hdl_mmol: float,
                           sbp: float, treated: int, smoker: int, diabetes: int) -> Tuple[int, float]:
    """
    Uses the point table shown in the CCS FRS tool PDF (mmol/L). :contentReference[oaicite:5]{index=5}
    Returns (points, 10-year risk %).
    """
    sex = sex.lower()
    if sex not in ["male", "female"]:
        raise ValueError("sex must be male/female")

    # AGE points
    # Table ranges: 30-34, 35-39, ..., 70-74, 75+
    def age_pts():
        if age < 30:
            return 0
        if 30 <= age <= 34:
            return 0 if sex == "male" else 0
        if 35 <= age <= 39:
            return 2 if sex == "male" else 2
        if 40 <= age <= 44:
            return 5 if sex == "male" else 4
        if 45 <= age <= 49:
            return 7 if sex == "male" else 5
        if 50 <= age <= 54:
            return 8 if sex == "male" else 7
        if 55 <= age <= 59:
            return 10 if sex == "male" else 8
        if 60 <= age <= 64:
            return 11 if sex == "male" else 9
        if 65 <= age <= 69:
            return 12 if sex == "male" else 10
        if 70 <= age <= 74:
            return 14 if sex == "male" else 11
        return 15 if sex == "male" else 12

    # HDL points
    def hdl_pts():
        if hdl_mmol > 1.6:
            return -2
        if 1.3 <= hdl_mmol <= 1.6:
            return -1
        if 1.2 <= hdl_mmol <= 1.29:
            return 0
        if 0.9 <= hdl_mmol <= 1.19:
            return 1
        return 2

    # Total cholesterol points (mmol/L)
    def tc_pts():
        if total_chol_mmol < 4.1:
            return 0
        if 4.1 <= total_chol_mmol <= 5.19:
            return 1
        if 5.2 <= total_chol_mmol <= 6.19:
            return 2 if sex == "male" else 3
        if 6.2 <= total_chol_mmol <= 7.2:
            return 3 if sex == "male" else 4
        return 4 if sex == "male" else 5

    # SBP points depend on treated and sex
    def sbp_pts():
        if treated == 1:
            if sbp < 120:
                return 0 if sex == "male" else -1
            if 120 <= sbp <= 129:
                return 2 if sex == "male" else 2
            if 130 <= sbp <= 139:
                return 3 if sex == "male" else 3
            if 140 <= sbp <= 149:
                return 4 if sex == "male" else 5
            if 150 <= sbp <= 159:
                return 4 if sex == "male" else 6
            return 5 if sex == "male" else 7
        else:
            if sbp < 120:
                return -2 if sex == "male" else -3
            if 120 <= sbp <= 129:
                return 0 if sex == "male" else 0
            if 130 <= sbp <= 139:
                return 1 if sex == "male" else 1
            if 140 <= sbp <= 149:
                return 2 if sex == "male" else 2
            if 150 <= sbp <= 159:
                return 2 if sex == "male" else 4
            return 3 if sex == "male" else 5

    points = age_pts() + hdl_pts() + tc_pts() + sbp_pts() + (4 if diabetes else 0) + (2 if smoker else 0)

    # Points -> 10-year risk % mapping from table
    # (Men/Women columns)
    risk_table_m = {
        -3: 0.9, -2: 1.1, -1: 1.4, 0: 1.6, 1: 1.9, 2: 2.3, 3: 2.8, 4: 3.3, 5: 3.9,
        6: 4.7, 7: 5.6, 8: 6.7, 9: 7.9, 10: 9.4, 11: 11.2, 12: 13.3, 13: 15.6,
        14: 18.4, 15: 21.6, 16: 25.3, 17: 29.4, 18: 30.1, 19: 30.1, 20: 30.1
    }
    risk_table_f = {
        -3: 0.9, -2: 0.9, -1: 1.0, 0: 1.2, 1: 1.5, 2: 1.7, 3: 2.0, 4: 2.4, 5: 2.8,
        6: 3.3, 7: 3.9, 8: 4.5, 9: 5.3, 10: 6.3, 11: 7.3, 12: 8.6, 13: 10.0,
        14: 11.7, 15: 13.7, 16: 15.9, 17: 18.5, 18: 21.5, 19: 24.8, 20: 27.5, 21: 30.1
    }

    if sex == "male":
        if points <= -3:
            risk = 0.9
        elif points >= 18:
            risk = 30.1
        else:
            risk = risk_table_m.get(points, 30.1)
    else:
        if points <= -3:
            risk = 0.9
        elif points >= 21:
            risk = 30.1
        else:
            risk = risk_table_f.get(points, 30.1)

    return int(points), float(risk)


# ----------------------------
# Main compute wrapper
# ----------------------------
def compute_scores(df: pd.DataFrame, chol_units: str, score2_region: str,
                   prevent_horizon: str, prevent_model_variant: str) -> pd.DataFrame:
    # Resolve columns
    col = {k: resolve_column(df, v) for k, v in DEFAULT_COL_MAP.items()}

    required = ["age", "sex", "sbp", "bp_tx", "total_c", "hdl_c", "statin", "dm", "smoking", "egfr", "bmi"]
    missing = [r for r in required if col[r] is None]
    if missing:
        raise ValueError(f"Missing required columns for PREVENT: {missing}")

    out_rows = []
    for i, r in df.iterrows():
        sex = normalize_sex(r[col["sex"]])
        if sex is None:
            out_rows.append({"row": i, "error": "Invalid sex"})
            continue

        # Build normalized row dict
        row = {
            "age": float(r[col["age"]]),
            "sex": sex,
            "sbp": float(r[col["sbp"]]),
            "bp_tx": to_binary(r[col["bp_tx"]]) or 0,
            "total_c": float(r[col["total_c"]]),
            "hdl_c": float(r[col["hdl_c"]]),
            "statin": to_binary(r[col["statin"]]) or 0,
            "dm": to_binary(r[col["dm"]]) or 0,
            "smoking": to_binary(r[col["smoking"]]) or 0,
            "egfr": float(r[col["egfr"]]),
            "bmi": float(r[col["bmi"]]),
        }

        # optional
        if col["hba1c"] is not None:
            row["hba1c"] = r[col["hba1c"]]
        if col["uacr"] is not None:
            row["uacr"] = r[col["uacr"]]
        # SDI not implemented from ZIP by default; allow user to provide 'sdi' column themselves if they add it
        if "sdi" in df.columns:
            row["sdi"] = r["sdi"]

        # PREVENT
        try:
            prevent = prevent_predict(row, chol_units=chol_units, horizon=prevent_horizon, model_variant=prevent_model_variant)
        except Exception as e:
            prevent = {"error": f"PREVENT failed: {e}"}

        # SCORE2 uses mmol/L for cholesterol
        tc_mmol, hdl_mmol = convert_chol_to_mmol(row["total_c"], row["hdl_c"], chol_units)
        try:
            score2_pct = score2_risk(
                region=score2_region,
                age=row["age"],
                sex="male" if sex == "male" else "female",
                smoker=row["smoking"],
                sbp=row["sbp"],
                diabetes=row["dm"],
                total_chol_mmol=tc_mmol,
                hdl_mmol=hdl_mmol,
            )
        except Exception as e:
            score2_pct = np.nan

        # Framingham 2008 points (mmol/L)
        try:
            fr_points, fr_risk = framingham_points_2008(
                sex="male" if sex == "male" else "female",
                age=row["age"],
                total_chol_mmol=tc_mmol,
                hdl_mmol=hdl_mmol,
                sbp=row["sbp"],
                treated=row["bp_tx"],
                smoker=row["smoking"],
                diabetes=row["dm"],
            )
        except Exception:
            fr_points, fr_risk = np.nan, np.nan

        base = {"row": i}
        # flatten prevent
        for k, v in prevent.items():
            base[f"prevent_{k}_{prevent_horizon}"] = v
        base["score2_10yr_percent"] = score2_pct
        base["frs2008_points"] = fr_points
        base["frs2008_10yr_percent"] = fr_risk

        out_rows.append(base)

    return pd.DataFrame(out_rows)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CVD Risk Scores", layout="wide")
st.title("CVD Risk Scores — PREVENT, SCORE2, Framingham 2008")

with st.expander("Expected CSV columns (minimum)"):
    st.markdown(
        """
**Minimum (PREVENT-ready):** age, sex, sbp, bp_tx, total_c, hdl_c, statin, dm, smoking, egfr, bmi  
**Optional:** hba1c, uacr, sdi (or zip if you later add SDI lookup)
        """
    )

left, right = st.columns([1, 1])

with left:
    chol_units = st.selectbox("Cholesterol units in your CSV", ["mg/dL", "mmol/L"], index=0)
    score2_region = st.selectbox("SCORE2 risk region (Europe model)", ["Low", "Moderate", "High", "Very High"], index=0)

with right:
    prevent_horizon = st.selectbox("PREVENT time horizon", ["10yr", "30yr"], index=0)
    prevent_model_variant = st.selectbox(
        "PREVENT model variant",
        ["base", "uacr", "hba1c", "sdi", "full"],
        index=0,
        help="If you pick 'uacr'/'hba1c'/'sdi'/'full', make sure your CSV includes those fields (or sdi).",
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("Compute scores"):
        with st.spinner("Computing… (first run downloads PREVENT coefficient tables)"):
            try:
                res = compute_scores(
                    df=df,
                    chol_units=chol_units,
                    score2_region=score2_region,
                    prevent_horizon=prevent_horizon,
                    prevent_model_variant=prevent_model_variant,
                )
                st.subheader("Results")
                st.dataframe(res, use_container_width=True)

                # download
                csv_bytes = res.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv_bytes, file_name="risk_scores_output.csv", mime="text/csv")
            except Exception as e:
                st.error(str(e))

st.caption(
    "Notes: SCORE2 is designed for European regions; calibration may be off outside Europe. "
    "PREVENT does not use race as an input; your sex still matters for the sex-specific equations."
)
