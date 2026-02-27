import numpy as np
import pandas as pd

# Coefficients from a published table of the ACC/AHA Pooled Cohort Equations parameters
# (White women, Black women, White men, Black men), including baseline survival and mean LP. :contentReference[oaicite:1]{index=1}
PCE_COEFS = {
    ("female", "white"): {
        "ln_age": -29.80,
        "ln_age_sq": 4.88,
        "ln_tc": 13.54,
        "ln_age_ln_tc": -3.11,
        "ln_hdl": -13.58,
        "ln_age_ln_hdl": 3.15,
        "ln_treated_sbp": 2.02,
        "ln_age_ln_treated_sbp": 0.00,
        "ln_untreated_sbp": 1.96,
        "ln_age_ln_untreated_sbp": 0.00,
        "smoker": 7.57,
        "ln_age_smoker": -1.67,
        "diabetes": 0.66,
        "mean_lp": -29.18,
        "baseline_survival": 0.9665,
    },
    ("female", "black"): {
        "ln_age": 17.114,
        "ln_age_sq": 0.0,
        "ln_tc": 0.94,
        "ln_age_ln_tc": 0.0,
        "ln_hdl": -18.92,
        "ln_age_ln_hdl": 4.475,
        "ln_treated_sbp": 29.291,
        "ln_age_ln_treated_sbp": -6.432,
        "ln_untreated_sbp": 27.82,
        "ln_age_ln_untreated_sbp": -6.087,
        "smoker": 0.691,
        "ln_age_smoker": 0.0,
        "diabetes": 0.874,
        "mean_lp": 86.61,
        "baseline_survival": 0.9533,
    },
    ("male", "white"): {
        "ln_age": 12.344,
        "ln_age_sq": 11.853,
        "ln_tc": 0.0,
        "ln_age_ln_tc": -2.664,
        "ln_hdl": -7.99,
        "ln_age_ln_hdl": 1.769,
        "ln_treated_sbp": 1.797,
        "ln_age_ln_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.764,
        "ln_age_ln_untreated_sbp": 0.0,
        "smoker": 7.837,
        "ln_age_smoker": -1.795,
        "diabetes": 0.0658,
        "mean_lp": 61.18,
        "baseline_survival": 0.9144,
    },
    ("male", "black"): {
        "ln_age": 2.469,
        "ln_age_sq": 0.302,
        "ln_tc": 0.0,
        "ln_age_ln_tc": 0.0,
        "ln_hdl": -0.307,
        "ln_age_ln_hdl": 0.0,
        "ln_treated_sbp": 1.916,
        "ln_age_ln_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.809,
        "ln_age_ln_untreated_sbp": 0.0,
        "smoker": 0.549,
        "ln_age_smoker": 0.0,
        "diabetes": 0.645,
        "mean_lp": 19.54,
        "baseline_survival": 0.8954,
    },
}

def _normalize_sex(x: str) -> str:
    x = str(x).strip().lower()
    if x in {"f", "female", "woman", "women"}:
        return "female"
    if x in {"m", "male", "man", "men"}:
        return "male"
    raise ValueError(f"Unknown sex value: {x}")

def _normalize_race_for_pce(x: str) -> str:
    """
    PCE is parameterized for Black vs White. Most implementations map 'Asian' -> 'white/other'.
    """
    x = str(x).strip().lower()
    if x in {"black", "african american", "aa"}:
        return "black"
    # treat asian/hispanic/other/unknown as white/other for classic PCE
    return "white"

def pce_ascvd_10y_row(row: pd.Series, default_sex="female", default_race="white") -> dict:
    sex = _normalize_sex(row.get("sex", default_sex))
    race = _normalize_race_for_pce(row.get("race", default_race))
    coefs = PCE_COEFS[(sex, race)]

    # Basic validation
    for col in ["age", "totchol", "hdl", "sbp", "bp_med", "smoker", "diabetes"]:
        if col not in row or pd.isna(row[col]):
            raise ValueError(f"Missing required field '{col}' in row index {row.name}")

    age = float(row["age"])
    tc = float(row["totchol"])
    hdl = float(row["hdl"])
    sbp = float(row["sbp"])
    bp_med = int(row["bp_med"])
    smoker = int(row["smoker"])
    diabetes = int(row["diabetes"])

    # Natural logs
    ln_age = np.log(age)
    ln_tc = np.log(tc)
    ln_hdl = np.log(hdl)
    ln_sbp = np.log(sbp)

    # Build linear predictor (LP)
    lp = 0.0
    lp += coefs["ln_age"] * ln_age
    lp += coefs["ln_age_sq"] * (ln_age ** 2)
    lp += coefs["ln_tc"] * ln_tc
    lp += coefs["ln_age_ln_tc"] * (ln_age * ln_tc)
    lp += coefs["ln_hdl"] * ln_hdl
    lp += coefs["ln_age_ln_hdl"] * (ln_age * ln_hdl)

    if bp_med == 1:
        lp += coefs["ln_treated_sbp"] * ln_sbp
        lp += coefs["ln_age_ln_treated_sbp"] * (ln_age * ln_sbp)
    else:
        lp += coefs["ln_untreated_sbp"] * ln_sbp
        lp += coefs["ln_age_ln_untreated_sbp"] * (ln_age * ln_sbp)

    lp += coefs["smoker"] * smoker
    lp += coefs["ln_age_smoker"] * (ln_age * smoker)
    lp += coefs["diabetes"] * diabetes

    # Risk = 1 - S0(10) ^ exp(LP - meanLP)
    risk_10y = 1.0 - (coefs["baseline_survival"] ** np.exp(lp - coefs["mean_lp"]))

    return {
        "pce_group_used": f"{sex}_{race}",
        "pce_lp": lp,
        "ascvd_10y_risk": float(risk_10y),            # 0..1
        "ascvd_10y_risk_pct": float(risk_10y * 100),  # percent
    }

def score_csv(input_csv_path: str, output_csv_path: str,
              default_sex="female", default_race="white") -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)

    # Compute row-wise
    results = df.apply(
        lambda r: pce_ascvd_10y_row(r, default_sex=default_sex, default_race=default_race),
        axis=1,
        result_type="expand",
    )

    out = pd.concat([df, results], axis=1)
    out.to_csv(output_csv_path, index=False)
    return out

if __name__ == "__main__":
    # Example:
    # score_csv("my_health_metrics.csv", "scored_metrics.csv", default_sex="female", default_race="white")
    pass
