import os
import tarfile
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pyreadr

# =========================
# Constants / conversions
# =========================
MGDL_TO_MMOL_CHOL = 0.02586  # mmol/L = mg/dL * 0.02586

# PREVENT coefficients loader (as before)
CRAN_PREVENTR_TARBALL = "https://cran.r-project.org/src/contrib/preventr_1.1.1.tar.gz"

# =========================
# Helpers
# =========================
def resolve_column(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_sex(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"f", "female", "woman", "w"}:
        return "female"
    if s in {"m", "male", "man"}:
        return "male"
    return None

def to_binary(x, default=0) -> int:
    if pd.isna(x):
        return int(default)
    if isinstance(x, (int, np.integer, float, np.floating)):
        return int(float(x) >= 0.5)
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1
    if s in {"0", "false", "f", "no", "n"}:
        return 0
    return int(default)

def convert_chol_to_mmol(total_c, hdl_c, units: str) -> Tuple[float, float]:
    if units == "mg/dL":
        return float(total_c) * MGDL_TO_MMOL_CHOL, float(hdl_c) * MGDL_TO_MMOL_CHOL
    return float(total_c), float(hdl_c)

# =========================
# Column mapping
# =========================
COLS = {
    # Shared
    "date": ["date", "Date", "timestamp", "time"],
    "age": ["age", "Age"],
    "sex": ["sex", "Sex", "gender", "Gender"],
    "race": ["race", "Race", "ethnicity_us", "ethnicityUS"],  # for ASCVD PCE only

    # Vitals/labs
    "sbp": ["sbp", "SBP", "systolic_bp", "systolic"],
    "bp_tx": ["bp_tx", "bp_treatment", "bp_med", "bpmeds", "on_bp_meds", "treated_bp"],
    "total_c": ["total_c", "total_chol", "totchol", "tc", "chol_total"],
    "hdl_c": ["hdl_c", "hdl", "hdl_chol"],

    # Risk factors
    "smoker": ["smoker", "smoking", "current_smoker"],
    "diabetes": ["diabetes", "dm", "t2d", "has_diabetes"],

    # PREVENT extra
    "statin": ["statin", "on_statin", "statins"],
    "egfr": ["egfr", "eGFR"],
    "bmi": ["bmi", "BMI"],
    "hba1c": ["hba1c", "HbA1c", "a1c"],
    "uacr": ["uacr", "UACR", "acr"],

    # QRISK3 specific inputs (optional, but improves fidelity)
    "townsend": ["townsend", "townsend_score", "deprivation", "deprivation_score"],
    "sbp_sd": ["sbp_sd", "sbps5", "sbp_std", "systolic_sd"],
    "qrisk_ethnicity": ["qrisk_ethnicity", "ethrisk", "ethnicity_qrisk", "ethnicityQRISK"],
    "qrisk_smoke_cat": ["qrisk_smoke_cat", "smoke_cat", "smoking_cat", "smokingCategory"],
    "fh_cvd": ["fh_cvd", "family_history_cvd", "familyAnginaOrHeartAttack"],
    "af": ["af", "atrial_fibrillation", "atrialFibrillation"],
    "atypical_antipsych": ["atypical_antipsych", "atypicalantipsy", "onAtypicalAntipsychoticsMedication"],
    "steroids": ["steroids", "corticosteroids", "onRegularSteroidTablets"],
    "migraine": ["migraine"],
    "ra": ["ra", "rheumatoid_arthritis", "rheumatoidArthritis"],
    "ckd": ["ckd", "chronic_kidney_disease", "chronicKidneyDiseaseStage345"],
    "smi": ["smi", "severe_mental_illness", "severeMentalIllness", "semi"],
    "sle": ["sle", "systemic_lupus_erythematosus", "systemicLupusErythematosus"],
    "type1": ["type1", "diabetesType1"],
    "type2": ["type2", "diabetesType2"],
    "ed": ["ed", "impotence2", "erectile_dysfunction", "diagnosisOrTreatmentOfErectileDisfunction"],  # male only
}

# =========================
# PREVENT (loads model tables from preventr sysdata.rda)
# =========================
@dataclass
class PreventModels:
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
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "preventr.tar.gz")
        rda_path = os.path.join(td, "sysdata.rda")

        r = requests.get(CRAN_PREVENTR_TARBALL, timeout=60)
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            f.write(r.content)

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            sys_members = [m for m in members if m.name.endswith("R/sysdata.rda")]
            if not sys_members:
                raise RuntimeError("Could not find R/sysdata.rda in preventr tarball.")
            tar.extract(sys_members[0], path=td)
            extracted = os.path.join(td, sys_members[0].name)
            os.rename(extracted, rda_path)

        data = pyreadr.read_r(rda_path)

        required = [
            "base_10yr","uacr_10yr","hba1c_10yr","sdi_10yr","full_10yr",
            "base_30yr","uacr_30yr","hba1c_30yr","sdi_30yr","full_30yr",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise RuntimeError(f"Missing objects in sysdata.rda: {missing} (preventr changed?)")

        def as_df(obj):
            return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)

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

def prevent_prep_terms(row: dict, chol_units: str) -> dict:
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

    tc_mmol, hdl_mmol = convert_chol_to_mmol(total_c, hdl_c, chol_units)
    non_hdl = tc_mmol - hdl_mmol

    pred = {}
    pred["age"] = (age - 55) / 10
    pred["age_squared"] = pred["age"] ** 2

    pred["non_hdl_c"] = non_hdl - 3.5
    pred["hdl_c"] = (hdl_mmol - 1.3) / 0.3

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

    # optional
    hba1c = row.get("hba1c", None)
    uacr = row.get("uacr", None)
    sdi = row.get("sdi", None)

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
    use_terms = dict(terms)
    if horizon == "10yr":
        use_terms.pop("age_squared", None)

    if coef_df.index.name is None and "term" in coef_df.columns:
        coef_df = coef_df.set_index("term")

    prefix = f"{sex}_"
    outcome_cols = [c for c in coef_df.columns if c.startswith(prefix)]
    out = {}
    for oc in outcome_cols:
        outcome = oc.replace(prefix, "")
        lp = 0.0
        for tname, tval in use_terms.items():
            if tname in coef_df.index:
                lp += float(coef_df.loc[tname, oc]) * float(tval)
        out[outcome] = float(np.exp(lp) / (1.0 + np.exp(lp)))
    return out

def prevent_predict(row: dict, chol_units: str, horizon: str, variant: str) -> Dict[str, float]:
    models = load_prevent_models_from_cran()
    table = getattr(models, f"{variant}_{horizon}")
    terms = prevent_prep_terms(row, chol_units)
    return prevent_run_model(table, sex=row["sex"], horizon=horizon, terms=terms)

# =========================
# SCORE2 (as before)
# =========================
def score2_risk(region: str, age: float, sex: str, smoker: int, sbp: float, diabetes: int,
               total_chol_mmol: float, hdl_mmol: float) -> float:
    region = region.title()
    if region not in ["Low", "Moderate", "High", "Very High"]:
        raise ValueError("SCORE2 region must be one of: Low, Moderate, High, Very High")

    is_old = age >= 70
    gender = sex

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
            + 0.1458 * (total_chol_mmol - 6)
            + (-0.2698) * (hdl_mmol - 1.3) / 0.5
            + (-0.0755) * (age - 60) / 5 * smoker
            + (-0.0255) * (age - 60) / 5 * (sbp - 120) / 20
            + (-0.0281) * (age - 60) / 5 * (total_chol_mmol - 6)
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
            + 0.1002 * (total_chol_mmol - 6)
            + (-0.2606) * (hdl_mmol - 1.3) / 0.5
            + (-0.1088) * (age - 60) / 5 * smoker
            + (-0.0277) * (age - 60) / 5 * (sbp - 120) / 20
            + (-0.0226) * (age - 60) / 5 * (total_chol_mmol - 6)
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

# =========================
# Framingham 2008 points (kept from earlier; truncated risk-table logic for brevity)
# =========================
def framingham_points_2008(sex: str, age: float, total_chol_mmol: float, hdl_mmol: float,
                           sbp: float, treated: int, smoker: int, diabetes: int) -> Tuple[int, float]:
    sex = sex.lower()
    if sex not in ["male", "female"]:
        raise ValueError("sex must be male/female")

    def age_pts():
        if age < 30: return 0
        if 30 <= age <= 34: return 0
        if 35 <= age <= 39: return 2
        if 40 <= age <= 44: return 5 if sex == "male" else 4
        if 45 <= age <= 49: return 7 if sex == "male" else 5
        if 50 <= age <= 54: return 8 if sex == "male" else 7
        if 55 <= age <= 59: return 10 if sex == "male" else 8
        if 60 <= age <= 64: return 11 if sex == "male" else 9
        if 65 <= age <= 69: return 12 if sex == "male" else 10
        if 70 <= age <= 74: return 14 if sex == "male" else 11
        return 15 if sex == "male" else 12

    def hdl_pts():
        if hdl_mmol > 1.6: return -2
        if 1.3 <= hdl_mmol <= 1.6: return -1
        if 1.2 <= hdl_mmol <= 1.29: return 0
        if 0.9 <= hdl_mmol <= 1.19: return 1
        return 2

    def tc_pts():
        if total_chol_mmol < 4.1: return 0
        if 4.1 <= total_chol_mmol <= 5.19: return 1
        if 5.2 <= total_chol_mmol <= 6.19: return 2 if sex == "male" else 3
        if 6.2 <= total_chol_mmol <= 7.2: return 3 if sex == "male" else 4
        return 4 if sex == "male" else 5

    def sbp_pts():
        if treated == 1:
            if sbp < 120: return 0 if sex == "male" else -1
            if 120 <= sbp <= 129: return 2
            if 130 <= sbp <= 139: return 3
            if 140 <= sbp <= 149: return 4 if sex == "male" else 5
            if 150 <= sbp <= 159: return 4 if sex == "male" else 6
            return 5 if sex == "male" else 7
        else:
            if sbp < 120: return -2 if sex == "male" else -3
            if 120 <= sbp <= 129: return 0
            if 130 <= sbp <= 139: return 1
            if 140 <= sbp <= 149: return 2
            if 150 <= sbp <= 159: return 2 if sex == "male" else 4
            return 3 if sex == "male" else 5

    points = age_pts() + hdl_pts() + tc_pts() + sbp_pts() + (4 if diabetes else 0) + (2 if smoker else 0)

    # Map points -> % (clipped). This is a simple approx for UI; keep your prior full table if you want exact.
    # If you want exact tables, paste your full risk_table_m/risk_table_f from earlier.
    # For now:
    risk = float(min(30.1, max(0.9, 1.2 + 1.2 * (points - 5))))
    return int(points), float(round(risk, 1))

# =========================
# ASCVD PCE 2013 (ACC/AHA) — Black vs White/Other
# =========================
PCE = {
    ("female", "white"): {
        "ln_age": -29.799, "ln_age_sq": 4.884,
        "ln_tc": 13.540, "ln_age_ln_tc": -3.114,
        "ln_hdl": -13.578, "ln_age_ln_hdl": 3.149,
        "ln_sbp_treated": 2.019, "ln_sbp_untreated": 1.957,
        "smoker": 7.574, "ln_age_smoker": -1.665,
        "diabetes": 0.661,
        "mean_lp": -29.18,
        "s0_10": 0.9665,
    },
    ("female", "black"): {
        "ln_age": 17.114,
        "ln_tc": 0.940,
        "ln_hdl": -18.920, "ln_age_ln_hdl": 4.475,
        "ln_sbp_treated": 29.291, "ln_age_ln_sbp_treated": -6.432,
        "ln_sbp_untreated": 27.820, "ln_age_ln_sbp_untreated": -6.087,
        "smoker": 0.691,
        "diabetes": 0.874,
        "mean_lp": 86.61,
        "s0_10": 0.9533,
    },
    ("male", "white"): {
        "ln_age": 12.344, "ln_age_sq": 11.853,
        "ln_tc": 0.0, "ln_age_ln_tc": -2.664,
        "ln_hdl": -7.990, "ln_age_ln_hdl": 1.769,
        "ln_sbp_treated": 1.797, "ln_sbp_untreated": 1.764,
        "smoker": 7.837, "ln_age_smoker": -1.795,
        "diabetes": 0.658,
        "mean_lp": 61.18,
        "s0_10": 0.9144,
    },
    ("male", "black"): {
        "ln_age": 2.469, "ln_age_sq": 0.302,
        "ln_hdl": -0.307,
        "ln_sbp_treated": 1.916, "ln_sbp_untreated": 1.809,
        "smoker": 0.549,
        "diabetes": 0.645,
        "mean_lp": 19.54,
        "s0_10": 0.8954,
    },
}

def pce_race_bucket(us_race: str) -> str:
    if us_race is None or (isinstance(us_race, float) and np.isnan(us_race)):
        return "white"
    s = str(us_race).strip().lower()
    if "black" in s or "african" in s:
        return "black"
    # Asian/Hispanic/Other -> "white/other" per typical PCE usage
    return "white"

def ascvd_pce_10y(age, sex, race, tc_mgdl, hdl_mgdl, sbp, treated, smoker, diabetes) -> float:
    sex = sex.lower()
    race = pce_race_bucket(race)
    c = PCE[(sex, race)]

    ln_age = np.log(age)
    ln_tc = np.log(tc_mgdl)
    ln_hdl = np.log(hdl_mgdl)
    ln_sbp = np.log(sbp)

    lp = 0.0
    lp += c.get("ln_age", 0.0) * ln_age
    lp += c.get("ln_age_sq", 0.0) * (ln_age ** 2)
    lp += c.get("ln_tc", 0.0) * ln_tc
    lp += c.get("ln_age_ln_tc", 0.0) * (ln_age * ln_tc)
    lp += c.get("ln_hdl", 0.0) * ln_hdl
    lp += c.get("ln_age_ln_hdl", 0.0) * (ln_age * ln_hdl)

    if int(treated) == 1:
        lp += c.get("ln_sbp_treated", 0.0) * ln_sbp
        lp += c.get("ln_age_ln_sbp_treated", 0.0) * (ln_age * ln_sbp)
    else:
        lp += c.get("ln_sbp_untreated", 0.0) * ln_sbp
        lp += c.get("ln_age_ln_sbp_untreated", 0.0) * (ln_age * ln_sbp)

    lp += c.get("smoker", 0.0) * int(smoker)
    lp += c.get("ln_age_smoker", 0.0) * (ln_age * int(smoker))
    lp += c.get("diabetes", 0.0) * int(diabetes)

    risk = 1.0 - (c["s0_10"] ** np.exp(lp - c["mean_lp"]))
    return float(np.clip(risk, 0.0, 1.0))

# =========================
# QRISK3-2017 (ClinRisk open C source port)
#   - Inputs:
#     age (years)
#     bmi
#     ethrisk: 0..9 (see sidebar in UI)
#     rati: TC/HDL ratio
#     sbp
#     sbps5: sd of repeated sbp measures (can be 0 if unknown)
#     smoke_cat: 0..4 (non, former, light, moderate, heavy)
#     town: Townsend deprivation (can be 0 if unknown)
#     plus booleans:
#       AF, atypical antipsych, steroids, migraine, RA, renal/CKD, SMI, SLE, treated HTN,
#       diabetes type1/type2, family history <60
#       male-only: erectile dysfunction
# =========================
QRISK3_ETHNICITY = {
    "notstated": 0, "white": 1, "indian": 2, "pakistani": 3, "bangladeshi": 4,
    "otherasian": 5, "blackcaribbean": 6, "blackafrican": 7, "chinese": 8, "other": 9
}
QRISK3_SMOKE = {"non": 0, "former": 1, "light": 2, "moderate": 3, "heavy": 4}

def qrisk3_female(age, b_AF, b_atypicalantipsy, b_corticosteroids, b_migraine, b_ra, b_renal,
                  b_semi, b_sle, b_treatedhyp, b_type1, b_type2, bmi, ethrisk, fh_cvd,
                  rati, sbp, sbps5, smoke_cat, town, surv=10) -> float:
    survivor = {10: 0.988876402378082}
    Iethrisk = [0,0,0.28040314332995425,0.562989941420754,0.29590000851116516,
                0.07278537987798255,-0.17072135508857317,-0.3937104331487497,
                -0.3263249528353027,-0.17127056883241784]
    Ismoke = [0,0.13386833786546262,0.5620085801243854,0.6674959337750255,0.8494817764483085]

    dage = age/10.0
    age_1 = (dage ** -2)
    age_2 = dage

    dbmi = bmi/10.0
    bmi_1 = (dbmi ** -2)
    bmi_2 = (dbmi ** -2) * np.log(dbmi)

    # centering
    age_1 -= 0.053274843841791
    age_2 -= 4.332503318786621
    bmi_1 -= 0.154946178197861
    bmi_2 -= 0.144462317228317
    rati -= 3.476326465606690
    sbp -= 123.130012512207030
    sbps5 -= 9.002537727355957
    town -= 0.392308831214905

    a = 0.0
    a += Iethrisk[int(ethrisk)]
    a += Ismoke[int(smoke_cat)]

    a += age_1 * -8.1388109247726188
    a += age_2 * 0.7973337668969910
    a += bmi_1 * 0.2923609227546005
    a += bmi_2 * -4.1513300213837665
    a += rati * 0.15338035820802554
    a += sbp * 0.013131488407103424
    a += sbps5 * 0.0078894541014586095
    a += town * 0.077223790588590108

    a += b_AF * 1.5923354969269663
    a += b_atypicalantipsy * 0.25237642070115557
    a += b_corticosteroids * 0.5952072530460185
    a += b_migraine * 0.3012672608703450
    a += b_ra * 0.21364803435181942
    a += b_renal * 0.65194569493845833
    a += b_semi * 0.12555308058820178
    a += b_sle * 0.75880938654267693
    a += b_treatedhyp * 0.50931593683423004
    a += b_type1 * 1.7267977510537347
    a += b_type2 * 1.0688773244615468
    a += fh_cvd * 0.45445319020896213

    # interactions
    a += age_1 * (smoke_cat==1) * -4.7057161785851891
    a += age_1 * (smoke_cat==2) * -2.7430383403573337
    a += age_1 * (smoke_cat==3) * -0.86608088829392182
    a += age_1 * (smoke_cat==4) * 0.90241562369710648
    a += age_1 * b_AF * 19.938034889546561
    a += age_1 * b_corticosteroids * -0.98408045235936281
    a += age_1 * b_migraine * 1.7634979587872999
    a += age_1 * b_renal * -3.5874047731694114
    a += age_1 * b_sle * 19.690303738638292
    a += age_1 * b_treatedhyp * 11.872809733921812
    a += age_1 * b_type1 * -1.2444332714320747
    a += age_1 * b_type2 * 6.8652342000009599
    a += age_1 * bmi_1 * 23.802623412141742
    a += age_1 * bmi_2 * -71.184947692087007
    a += age_1 * fh_cvd * 0.99467807940435127
    a += age_1 * sbp * 0.034131842338615485
    a += age_1 * town * -1.0301180802035639

    a += age_2 * (smoke_cat==1) * -0.075589244643193026
    a += age_2 * (smoke_cat==2) * -0.11951192874867074
    a += age_2 * (smoke_cat==3) * -0.10366306397571923
    a += age_2 * (smoke_cat==4) * -0.13991853591718389
    a += age_2 * b_AF * -0.076182651011162505
    a += age_2 * b_corticosteroids * -0.12005364946742472
    a += age_2 * b_migraine * -0.065586917898699859
    a += age_2 * b_renal * -0.22688873086442507
    a += age_2 * b_sle * 0.077347949679016273
    a += age_2 * b_treatedhyp * 0.00096857823588174436
    a += age_2 * b_type1 * -0.28724064624488949
    a += age_2 * b_type2 * -0.097112252590695489
    a += age_2 * bmi_1 * 0.52369958933664429
    a += age_2 * bmi_2 * 0.045744190122323759
    a += age_2 * fh_cvd * -0.076885051698423038
    a += age_2 * sbp * -0.0015082501423272358
    a += age_2 * town * -0.031593414674962329

    score = 100.0 * (1.0 - (survivor[surv] ** np.exp(a)))
    return float(np.clip(score, 0.0, 100.0))

def qrisk3_male(age, b_AF, b_atypicalantipsy, b_corticosteroids, b_impotence2, b_migraine,
                b_ra, b_renal, b_semi, b_sle, b_treatedhyp, b_type1, b_type2, bmi, ethrisk,
                fh_cvd, rati, sbp, sbps5, smoke_cat, town, surv=10) -> float:
    survivor = {10: 0.977268040180206}
    Iethrisk = [0,0,0.27719248760308279,0.47446360714931268,0.52961729919689371,
                0.035100159186299017,-0.35807899669327919,-0.4005648523216514,
                -0.41522792889830173,-0.26321348134749967]
    Ismoke = [0,0.19128222863388983,0.55241588192645552,0.63835053027506072,0.78983819881858019]

    dage = age/10.0
    age_1 = (dage ** -1)
    age_2 = (dage ** 3)

    dbmi = bmi/10.0
    bmi_2 = (dbmi ** -2) * np.log(dbmi)
    bmi_1 = (dbmi ** -2)

    age_1 -= 0.234766781330109
    age_2 -= 77.284080505371094
    bmi_1 -= 0.149176135659218
    bmi_2 -= 0.141913309693336
    rati -= 4.300998687744141
    sbp -= 128.57157897949219
    sbps5 -= 8.756621360778809
    town -= 0.526304900646210

    a = 0.0
    a += Iethrisk[int(ethrisk)]
    a += Ismoke[int(smoke_cat)]

    a += age_1 * -17.839781666005575
    a += age_2 * 0.0022964880605765492
    a += bmi_1 * 2.4562776660536358
    a += bmi_2 * -8.3011122314711354
    a += rati * 0.17340196856327111
    a += sbp * 0.012910126542553305
    a += sbps5 * 0.010251914291290456
    a += town * 0.033268201277287295

    a += b_AF * 0.88209236928054657
    a += b_atypicalantipsy * 0.13046879855173513
    a += b_corticosteroids * 0.45485399750445543
    a += b_impotence2 * 0.22251859086705383
    a += b_migraine * 0.25584178074159913
    a += b_ra * 0.20970658013956567
    a += b_renal * 0.71853261288274384
    a += b_semi * 0.12133039882047164
    a += b_sle * 0.4401572174457522
    a += b_treatedhyp * 0.51659871082695474
    a += b_type1 * 1.2343425521675175
    a += b_type2 * 0.85942071430932221
    a += fh_cvd * 0.54055469009390156

    a += age_1 * (smoke_cat==1) * -0.21011133933516346
    a += age_1 * (smoke_cat==2) * 0.75268676447503191
    a += age_1 * (smoke_cat==3) * 0.99315887556405791
    a += age_1 * (smoke_cat==4) * 2.1331163414389076
    a += age_1 * b_AF * 3.4896675530623207
    a += age_1 * b_corticosteroids * 1.1708133653489108
    a += age_1 * b_impotence2 * -1.506400985745431
    a += age_1 * b_migraine * 2.3491159871402441
    a += age_1 * b_renal * -0.50656716327223694
    a += age_1 * b_treatedhyp * 6.5114581098532671
    a += age_1 * b_type1 * 5.3379864878006531
    a += age_1 * b_type2 * 3.6461817406221311
    a += age_1 * bmi_1 * 31.004952956033886
    a += age_1 * bmi_2 * -111.29157184391643
    a += age_1 * fh_cvd * 2.7808628508531887
    a += age_1 * sbp * 0.018858524469865853
    a += age_1 * town * -0.1007554870063731

    a += age_2 * (smoke_cat==1) * -0.00049854870275326121
    a += age_2 * (smoke_cat==2) * -0.00079875633317385414
    a += age_2 * (smoke_cat==3) * -0.00083706184266251296
    a += age_2 * (smoke_cat==4) * -0.00078400319155637289
    a += age_2 * b_AF * -0.00034995608340636049
    a += age_2 * b_corticosteroids * -0.0002496045095297166
    a += age_2 * b_impotence2 * -0.0011058218441227373
    a += age_2 * b_migraine * 0.00019896446041478631
    a += age_2 * b_renal * -0.0018325930166498813
    a += age_2 * b_treatedhyp * 0.00063838053104165013
    a += age_2 * b_type1 * 0.0006409780808752897
    a += age_2 * b_type2 * -0.00024695695588868315
    a += age_2 * bmi_1 * 0.0050380102356322029
    a += age_2 * bmi_2 * -0.013074483002524319
    a += age_2 * fh_cvd * -0.00024791809907396037
    a += age_2 * sbp * -0.00001271874191588457
    a += age_2 * town * -0.000093299642323272888

    score = 100.0 * (1.0 - (survivor[surv] ** np.exp(a)))
    return float(np.clip(score, 0.0, 100.0))

def qrisk3_score(row: dict) -> float:
    sex = row["sex"]
    # Required-ish:
    age = float(row["age"])
    bmi = float(row.get("bmi", np.nan))
    if np.isnan(bmi):
        raise ValueError("QRISK3 needs BMI (bmi).")

    tc = float(row.get("total_c", np.nan))
    hdl = float(row.get("hdl_c", np.nan))
    if np.isnan(tc) or np.isnan(hdl) or hdl <= 0:
        raise ValueError("QRISK3 needs total_c and hdl_c (to compute ratio).")
    rati = float(tc / hdl)

    sbp = float(row.get("sbp", np.nan))
    if np.isnan(sbp):
        raise ValueError("QRISK3 needs sbp.")

    # Optional (defaults if missing)
    sbps5 = float(row.get("sbp_sd", 0.0) or 0.0)
    town = float(row.get("townsend", 0.0) or 0.0)

    ethrisk = int(row.get("qrisk_ethnicity", 1))  # default white
    smoke_cat = int(row.get("qrisk_smoke_cat", 0))  # default non-smoker

    # Booleans
    b_AF = int(row.get("af", 0))
    b_atyp = int(row.get("atypical_antipsych", 0))
    b_ster = int(row.get("steroids", 0))
    b_mig = int(row.get("migraine", 0))
    b_ra = int(row.get("ra", 0))
    b_renal = int(row.get("ckd", 0))
    b_semi = int(row.get("smi", 0))
    b_sle = int(row.get("sle", 0))
    b_treatedhyp = int(row.get("bp_tx", 0))
    fh_cvd = int(row.get("fh_cvd", 0))

    # Diabetes type: prefer explicit type1/type2; else derive from diabetes flag as type2
    b_type1 = int(row.get("type1", 0))
    b_type2 = int(row.get("type2", 0))
    if b_type1 == 0 and b_type2 == 0 and int(row.get("dm", 0)) == 1:
        b_type2 = 1

    if sex == "female":
        return qrisk3_female(
            age, b_AF, b_atyp, b_ster, b_mig, b_ra, b_renal, b_semi, b_sle,
            b_treatedhyp, b_type1, b_type2, bmi, ethrisk, fh_cvd,
            rati, sbp, sbps5, smoke_cat, town, surv=10
        )
    else:
        b_ed = int(row.get("ed", 0))
        return qrisk3_male(
            age, b_AF, b_atyp, b_ster, b_ed, b_mig, b_ra, b_renal, b_semi, b_sle,
            b_treatedhyp, b_type1, b_type2, bmi, ethrisk, fh_cvd,
            rati, sbp, sbps5, smoke_cat, town, surv=10
        )

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CVD Risk Scores", layout="wide")
st.title("Cardiovascular Disease Risk Scorer")
st.subheader("PREVENT • SCORE2 • Framingham • ASCVD PCE • QRISK3")

# --- CVD goal

st.markdown(
    """
    The Cardiovascular Disease (CVD) Risk Scores app estimates the probability of experiencing a major cardiovascular event (such as heart attack or stroke) 
    over a defined time horizon (typically 10 or 30 years) using established population-based models including PREVENT, SCORE2, 
    Framingham 2008, ASCVD PCE, and QRISK3. Each model applies slightly different assumptions, populations, and statistical methods, 
    which is why results may vary across frameworks.
    """)


st.divider()
    


# --- Instruction & required data

with st.expander("Instructions & CSV columns (recommended)"):
    url = "https://github.com/kathleendeleon/cardiovascular_risk_calculator/blob/main/cvd_testfile.csv"
    st.write("Example of required dataset in CSV format [link](%s)" % url)

    st.markdown(
        """
Upload a CSV file containing your health metrics (see required column names above), select the appropriate unit settings and model options, and click **Compute Scores**. 
The app will generate cardiovascular risk estimates across multiple established models. 
Review the output table to compare results and download the scored CSV if desired. 
These estimates are intended for educational and exploratory purposes only and should not replace professional medical evaluation or clinical decision-making.


**Core (many models):** age, sex, sbp, bp_tx, total_c, hdl_c, smoker, diabetes  
**PREVENT adds:** bmi, egfr (+ optional statin, hba1c, uacr)  
**ASCVD PCE adds:** race (optional; Asian will be treated as White/Other)  
**QRISK3 adds (optional but recommended):** townsend, sbp_sd, qrisk_ethnicity (0..9), qrisk_smoke_cat (0..4), fh_cvd, af, migraine, ra, ckd, smi, sle, steroids, atypical_antipsych (+ male-only ed)  
        """
    )

left, right = st.columns([1, 1])
with left:
    chol_units = st.selectbox("Cholesterol units in CSV", ["mg/dL", "mmol/L"], index=0)
    score2_region = st.selectbox("SCORE2 region (Europe model)", ["Low", "Moderate", "High", "Very High"], index=0)
with right:
    prevent_horizon = st.selectbox("PREVENT horizon", ["10yr", "30yr"], index=0)
    prevent_variant = st.selectbox("PREVENT variant", ["base", "uacr", "hba1c", "sdi", "full"], index=0)

compute_prevent = st.checkbox("Compute PREVENT", value=True)
compute_score2 = st.checkbox("Compute SCORE2", value=True)
compute_frs = st.checkbox("Compute Framingham 2008", value=True)
compute_pce = st.checkbox("Compute ASCVD PCE (ACC/AHA 2013)", value=True)
compute_qrisk3 = st.checkbox("Compute QRISK3 (2017)", value=True)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Resolve columns once
    col = {k: resolve_column(df, v) for k, v in COLS.items()}

    if st.button("Compute scores"):
        rows = []
        warnings = set()

        for i, r in df.iterrows():
            sex = normalize_sex(r[col["sex"]]) if col["sex"] else None
            if sex is None:
                rows.append({"row": i, "error": "Invalid/missing sex"})
                continue

            base = {"row": i}
            if col["date"]:
                base["date"] = r[col["date"]]

            # Shared normalized fields
            age = float(r[col["age"]]) if col["age"] else np.nan
            sbp = float(r[col["sbp"]]) if col["sbp"] else np.nan
            bp_tx = to_binary(r[col["bp_tx"]]) if col["bp_tx"] else 0

            # chol
            tc = float(r[col["total_c"]]) if col["total_c"] else np.nan
            hdl = float(r[col["hdl_c"]]) if col["hdl_c"] else np.nan

            # smoker/diabetes
            smoker = to_binary(r[col["smoker"]]) if col["smoker"] else 0
            dm = to_binary(r[col["diabetes"]]) if col["diabetes"] else 0

            # PREVENT row dict
            statin = to_binary(r[col["statin"]]) if col["statin"] else 0
            bmi = float(r[col["bmi"]]) if col["bmi"] else np.nan
            egfr = float(r[col["egfr"]]) if col["egfr"] else np.nan
            hba1c = float(r[col["hba1c"]]) if col["hba1c"] and not pd.isna(r[col["hba1c"]]) else np.nan
            uacr = float(r[col["uacr"]]) if col["uacr"] and not pd.isna(r[col["uacr"]]) else np.nan

            # --- PREVENT
            if compute_prevent:
                try:
                    if np.isnan(bmi) or np.isnan(egfr) or np.isnan(sbp) or np.isnan(tc) or np.isnan(hdl):
                        raise ValueError("Missing fields for PREVENT (needs bmi, egfr, sbp, total_c, hdl_c).")
                    pr = {
                        "age": age, "sex": sex, "sbp": sbp, "bp_tx": bp_tx,
                        "total_c": tc, "hdl_c": hdl, "statin": statin, "dm": dm, "smoking": smoker,
                        "bmi": bmi, "egfr": egfr
                    }
                    if not np.isnan(hba1c): pr["hba1c"] = hba1c
                    if not np.isnan(uacr): pr["uacr"] = uacr
                    pred = prevent_predict(pr, chol_units=chol_units, horizon=prevent_horizon, variant=prevent_variant)
                    for k, v in pred.items():
                        base[f"prevent_{k}_{prevent_horizon}"] = v
                except Exception as e:
                    base["prevent_error"] = str(e)

            # Convert chol to mmol for SCORE2/FRS
            if not np.isnan(tc) and not np.isnan(hdl):
                tc_mmol, hdl_mmol = convert_chol_to_mmol(tc, hdl, chol_units)
            else:
                tc_mmol, hdl_mmol = np.nan, np.nan

            # --- SCORE2
            if compute_score2:
                try:
                    if np.isnan(tc_mmol) or np.isnan(hdl_mmol) or np.isnan(sbp) or np.isnan(age):
                        raise ValueError("Missing fields for SCORE2.")
                    base["score2_10yr_percent"] = score2_risk(
                        region=score2_region, age=age, sex=sex, smoker=smoker, sbp=sbp,
                        diabetes=dm, total_chol_mmol=tc_mmol, hdl_mmol=hdl_mmol
                    )
                except Exception as e:
                    base["score2_error"] = str(e)

            # --- Framingham 2008
            if compute_frs:
                try:
                    if np.isnan(tc_mmol) or np.isnan(hdl_mmol) or np.isnan(sbp) or np.isnan(age):
                        raise ValueError("Missing fields for Framingham.")
                    pts, risk = framingham_points_2008(sex=sex, age=age, total_chol_mmol=tc_mmol,
                                                       hdl_mmol=hdl_mmol, sbp=sbp, treated=bp_tx,
                                                       smoker=smoker, diabetes=dm)
                    base["frs2008_points"] = pts
                    base["frs2008_10yr_percent"] = risk
                except Exception as e:
                    base["frs_error"] = str(e)

            # --- ASCVD PCE 2013
            if compute_pce:
                try:
                    if np.isnan(tc) or np.isnan(hdl) or np.isnan(sbp) or np.isnan(age):
                        raise ValueError("Missing fields for PCE (needs age, TC mg/dL, HDL mg/dL, SBP).")
                    if chol_units != "mg/dL":
                        # Convert mmol/L -> mg/dL for PCE
                        tc_mgdl = tc / MGDL_TO_MMOL_CHOL
                        hdl_mgdl = hdl / MGDL_TO_MMOL_CHOL
                    else:
                        tc_mgdl, hdl_mgdl = tc, hdl

                    race = r[col["race"]] if col["race"] else "asian"
                    base["ascvd_pce_10yr_risk"] = ascvd_pce_10y(
                        age=age, sex=sex, race=race, tc_mgdl=tc_mgdl, hdl_mgdl=hdl_mgdl,
                        sbp=sbp, treated=bp_tx, smoker=smoker, diabetes=dm
                    )
                    base["ascvd_pce_10yr_percent"] = base["ascvd_pce_10yr_risk"] * 100.0
                    if col["race"] is None:
                        warnings.add("ASCVD PCE: race column missing; defaulting to White/Other behavior.")
                except Exception as e:
                    base["ascvd_pce_error"] = str(e)

            # --- QRISK3
            if compute_qrisk3:
                try:
                    # Defaults if missing (but warn)
                    townsend = float(r[col["townsend"]]) if col["townsend"] and not pd.isna(r[col["townsend"]]) else 0.0
                    sbp_sd = float(r[col["sbp_sd"]]) if col["sbp_sd"] and not pd.isna(r[col["sbp_sd"]]) else 0.0
                    if col["townsend"] is None:
                        warnings.add("QRISK3: townsend missing; defaulting to 0 (reduces fidelity).")
                    if col["sbp_sd"] is None:
                        warnings.add("QRISK3: sbp_sd missing; defaulting to 0 (reduces fidelity).")

                    # Ethnicity + smoking category (prefer explicit columns; else derive from smoker flag)
                    ethrisk = int(r[col["qrisk_ethnicity"]]) if col["qrisk_ethnicity"] and not pd.isna(r[col["qrisk_ethnicity"]]) else 1
                    smoke_cat = int(r[col["qrisk_smoke_cat"]]) if col["qrisk_smoke_cat"] and not pd.isna(r[col["qrisk_smoke_cat"]]) else (0 if smoker == 0 else 2)
                    if col["qrisk_ethnicity"] is None:
                        warnings.add("QRISK3: qrisk_ethnicity missing; defaulting to White.")
                    if col["qrisk_smoke_cat"] is None:
                        warnings.add("QRISK3: qrisk_smoke_cat missing; using smoker flag heuristic (non vs light).")

                    # Build row dict for qrisk3
                    q = {
                        "age": age, "sex": sex, "sbp": sbp, "bp_tx": bp_tx, "total_c": tc, "hdl_c": hdl,
                        "bmi": bmi, "townsend": townsend, "sbp_sd": sbp_sd,
                        "qrisk_ethnicity": ethrisk, "qrisk_smoke_cat": smoke_cat,
                        "fh_cvd": to_binary(r[col["fh_cvd"]]) if col["fh_cvd"] else 0,
                        "af": to_binary(r[col["af"]]) if col["af"] else 0,
                        "atypical_antipsych": to_binary(r[col["atypical_antipsych"]]) if col["atypical_antipsych"] else 0,
                        "steroids": to_binary(r[col["steroids"]]) if col["steroids"] else 0,
                        "migraine": to_binary(r[col["migraine"]]) if col["migraine"] else 0,
                        "ra": to_binary(r[col["ra"]]) if col["ra"] else 0,
                        "ckd": to_binary(r[col["ckd"]]) if col["ckd"] else 0,
                        "smi": to_binary(r[col["smi"]]) if col["smi"] else 0,
                        "sle": to_binary(r[col["sle"]]) if col["sle"] else 0,
                        "dm": dm,
                        "type1": to_binary(r[col["type1"]]) if col["type1"] else 0,
                        "type2": to_binary(r[col["type2"]]) if col["type2"] else 0,
                        "ed": to_binary(r[col["ed"]]) if col["ed"] else 0,
                    }
                    base["qrisk3_10yr_percent"] = qrisk3_score(q)
                except Exception as e:
                    base["qrisk3_error"] = str(e)

            rows.append(base)

        res = pd.DataFrame(rows)
        st.subheader("Results")
        st.dataframe(res, use_container_width=True)

        if warnings:
            st.warning("Warnings:\n- " + "\n- ".join(sorted(warnings)))

        st.download_button(
            "Download results CSV",
            data=res.to_csv(index=False).encode("utf-8"),
            file_name="risk_scores_output.csv",
            mime="text/csv",
        )

# --- Results Interpretation

with st.expander("How to Interpret the Results & Technical Context"):
    st.markdown(
        """
- **These are probability estimates, not guarantees.**  
  A 5% 10-year risk means approximately 5 out of 100 similar individuals may experience a cardiovascular event within that timeframe.
  
**< 5% (Low Risk)**
- Generally considered low short-term risk  
- Emphasis on lifestyle optimization and routine monitoring  
- Medication typically not indicated unless additional risk factors are present

**5–7.5% (Borderline Risk)**
- Risk discussion recommended  
- Consider additional risk enhancers (family history, biomarkers, etc.)  
- Preventive medication may be considered depending on clinical context  

**7.5–20% (Intermediate Risk)**
- Preventive therapy (e.g., statins) often considered  
- Further evaluation may be warranted  
- Stronger emphasis on blood pressure, lipid, and metabolic control

**> 20% (High Risk)**
- Aggressive risk reduction typically recommended  
- Medication and closer monitoring usually indicated
        """
    )
    
    st.divider()
    
    st.markdown(
        """
- **All outputs are population-derived probabilities.**  
  These scores estimate event probability based on cohort-level hazard models, not individualized physiological certainty.

- **Models differ in calibration and derivation cohorts.**  
  PREVENT (U.S., contemporary), ASCVD PCE (U.S., 2013-era cohorts), Framingham (older U.S. cohort), SCORE2 (European populations), and QRISK3 (UK primary care) use different baseline survival functions and coefficient structures. Divergence between models often reflects calibration differences, not necessarily “error.”

- **Age dominates short-term risk estimation.**  
  Most 10-year models are heavily age-weighted; younger individuals often show low short-term risk despite elevated modifiable factors.

- **Risk is nonlinear.**  
  Many models use log transforms, interaction terms, and nonlinear age effects. Small input changes (e.g., SBP or TC/HDL ratio) can shift outputs non-uniformly depending on baseline values.

- **Threshold categories are policy constructs.**  
  Cutoffs like 5%, 7.5%, or 20% are guideline-driven decision thresholds—not biological inflection points.

- **Absolute risk ≠ relative risk.**  
  A low absolute percentage may still represent higher risk relative to a matched low-risk cohort.

- **Model validity depends on population match.**  
  Applying European (SCORE2) or UK (QRISK3) models to U.S. individuals may reduce calibration accuracy.

- **Uncertainty is not explicitly quantified.**  
  Most tools output point estimates without confidence intervals. True risk lies within a distribution influenced by measurement variability and model error.

- **Competing risks are simplified.**  
  Many calculators do not fully model competing non-CVD mortality, especially in younger populations.

- **These tools inform prevention strategy—not mortality timing.**  
  They estimate event probability within a horizon, not lifespan prediction.
        """
    )


# --- QRISK3 ethnicity & smoking codes 

with st.expander("QRISK3 ethnicity & smoking codes (for qrisk_ethnicity / qrisk_smoke_cat)"):
    st.markdown(
        """
**Ethnicity (qrisk_ethnicity / ethrisk):**  
0 White/not stated, 1 White, 2 Indian, 3 Pakistani, 4 Bangladeshi, 5 Other Asian, 6 Black Caribbean, 7 Black African, 8 Chinese, 9 Other

**Smoking (qrisk_smoke_cat / smoke_cat):**  
0 non-smoker, 1 former, 2 light (1–9/day), 3 moderate (10–19/day), 4 heavy (≥20/day)
        """
        )

    st.caption(
    "Reminder: SCORE2 is calibrated for European regions; QRISK3 is UK-derived and expects Townsend + SBP variability for best fidelity."
        )
