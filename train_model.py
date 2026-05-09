#!/usr/bin/env python3
"""
==============================================================
SEPSIS PREDICTION MODEL TRAINER
Project: Explainable Multimodal AI-based Sepsis Prediction
Dataset: PhysioNet MIMIC-III (PostgreSQL local)
==============================================================

Run this ONCE before starting the Flask app:
    cd Fyp-part1-main-main
    python train_model.py

It will:
  1. Pull structured data from your MIMIC-III PostgreSQL database
  2. Engineer clinical features (SIRS, qSOFA, SOFA components)
  3. Train 4 ML models and pick the best one (by F1-score)
  4. Save the model, scaler, feature names, and SHAP explainer
     to  models/saved_models/
"""

import os
import sys
import json
import pickle
import warnings
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE

import joblib
import shap


os.makedirs("logs", exist_ok=True)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────
# DATABASE CONFIG  ← change password / host if needed
# ─────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "mimiciii",
    "user":     "postgres",
    "password": "Fozia786",   # <── your PostgreSQL password
}

SAVE_DIR = "models/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────────
# MIMIC-III ITEM IDS  (standard PhysioNet item IDs)
# ─────────────────────────────────────────────────
VITAL_ITEM_IDS = {
    "heart_rate":       [211, 220045],
    "systolic_bp":      [51, 442, 455, 6701, 220179, 220050],
    "diastolic_bp":     [8368, 8440, 8441, 8555, 220180, 220051],
    "temperature":      [678, 679, 223761, 223762],   # °F ids → will convert
    "respiratory_rate": [618, 619, 220210, 224690],
    "spo2":             [646, 220277],
}

LAB_ITEM_IDS = {
    "wbc":        [51300, 51301],
    "lactate":    [50813],
    "creatinine": [50912],
    "platelets":  [51265],
    "bilirubin":  [50885],
    "glucose":    [50931, 50809],
    "sodium":     [50824, 50983],
    "potassium":  [50822, 50971],
    "bicarbonate":[50882],
    "pco2":       [50818],
    "po2":        [50821],
    "ph":         [50820],
}

# Sepsis ICD-9 codes
SEPSIS_ICD9 = (
    '99591', '99592',   # Sepsis / Severe sepsis (ICD-9-CM)
    '99500',            # Systemic inflammatory response syndrome (SIRS)
    '78552',            # Septic shock
    '03810', '03811', '03812', '03819',  # Streptococcal septicemia
    '02010', '02011', '02012', '02019',  # Staphylococcal septicemia
)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def get_connection():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    return conn


def run_query(sql: str, params=None) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Returns empty DF on error."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as exc:
        log.error(f"Query failed: {exc}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING FROM MIMIC-III
# ─────────────────────────────────────────────────────────────────────────────

def load_admissions(limit: int = 15000) -> pd.DataFrame:
    sql = f"""
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admission_type,
        a.ethnicity,
        a.hospital_expire_flag,
        EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 3600.0 AS los_hours,
        p.gender,
        CASE
            WHEN p.dob IS NULL THEN NULL
            ELSE DATE_PART('year', AGE(a.admittime, p.dob))
        END AS age
    FROM admissions a
    JOIN patients    p ON p.subject_id = a.subject_id
    WHERE a.dischtime IS NOT NULL
    LIMIT {limit}
    """
    df = run_query(sql)
    log.info(f"Admissions loaded: {len(df)} rows")
    return df


def load_sepsis_labels(hadm_ids: list) -> pd.DataFrame:
    """Return hadm_id -> has_sepsis (1/0) based on ICD-9 diagnosis codes."""
    if not hadm_ids:
        return pd.DataFrame(columns=["hadm_id", "has_sepsis"])

    placeholders = ",".join(["%s"] * len(hadm_ids))
    sql = f"""
    SELECT DISTINCT hadm_id
    FROM diagnoses_icd
    WHERE hadm_id IN ({placeholders})
      AND icd9_code IN ({",".join(["%s"]*len(SEPSIS_ICD9))})
    """
    sepsis_hadm = run_query(sql, tuple(hadm_ids) + SEPSIS_ICD9)

    if sepsis_hadm.empty:
        # Fall-back: text-based search on diagnosis field in admissions
        log.warning("ICD-9 sepsis query returned 0 rows – trying text fallback")
        sql2 = f"""
        SELECT DISTINCT hadm_id
        FROM admissions
        WHERE hadm_id IN ({placeholders})
          AND LOWER(diagnosis) ~ 'sepsis|septic|sirs'
        """
        sepsis_hadm = run_query(sql2, tuple(hadm_ids))

    labels = pd.DataFrame({"hadm_id": hadm_ids})
    labels["has_sepsis"] = labels["hadm_id"].isin(
        sepsis_hadm["hadm_id"].tolist()
    ).astype(int)
    log.info(
        f"Sepsis labels: {labels['has_sepsis'].sum()} positive / "
        f"{len(labels)} total ({labels['has_sepsis'].mean()*100:.1f}%)"
    )
    return labels


def load_vitals(hadm_ids: list, limit: int = 50000) -> pd.DataFrame:
    """Aggregate latest vitals per admission."""
    all_item_ids = []
    for ids in VITAL_ITEM_IDS.values():
        all_item_ids.extend(ids)

    if not hadm_ids or not all_item_ids:
        return pd.DataFrame()

    hadm_ph  = ",".join(["%s"] * len(hadm_ids))
    item_ph  = ",".join(["%s"] * len(all_item_ids))

    sql = f"""
    SELECT
        ce.hadm_id,
        ce.itemid,
        ce.valuenum
    FROM chartevents ce
    WHERE ce.hadm_id   IN ({hadm_ph})
      AND ce.itemid    IN ({item_ph})
      AND ce.valuenum  IS NOT NULL
      AND ce.valuenum   > 0
      AND ce.error     IS DISTINCT FROM 1
    LIMIT {limit}
    """
    raw = run_query(sql, tuple(hadm_ids) + tuple(all_item_ids))
    if raw.empty:
        log.warning("No chartevents vitals found.")
        return pd.DataFrame()

    # Map itemid → vital name
    id_to_name = {}
    for name, ids in VITAL_ITEM_IDS.items():
        for i in ids:
            id_to_name[i] = name
    raw["vital_name"] = raw["itemid"].map(id_to_name)

    # Convert °F → °C for temperature
    temp_mask = raw["vital_name"] == "temperature"
    raw.loc[temp_mask, "valuenum"] = (
        raw.loc[temp_mask, "valuenum"].apply(
            lambda v: (v - 32) * 5 / 9 if v > 50 else v
        )
    )

    # Aggregate: use MEDIAN per admission per vital
    pivoted = (
        raw.groupby(["hadm_id", "vital_name"])["valuenum"]
        .median()
        .unstack("vital_name")
        .reset_index()
    )
    log.info(f"Vitals aggregated: {len(pivoted)} admissions × {len(pivoted.columns)-1} vitals")
    return pivoted


def load_labs(hadm_ids: list, limit: int = 60000) -> pd.DataFrame:
    """Aggregate median lab values per admission."""
    all_item_ids = []
    for ids in LAB_ITEM_IDS.values():
        all_item_ids.extend(ids)

    if not hadm_ids or not all_item_ids:
        return pd.DataFrame()

    hadm_ph = ",".join(["%s"] * len(hadm_ids))
    item_ph = ",".join(["%s"] * len(all_item_ids))

    sql = f"""
    SELECT
        le.hadm_id,
        le.itemid,
        le.valuenum
    FROM labevents le
    WHERE le.hadm_id  IN ({hadm_ph})
      AND le.itemid   IN ({item_ph})
      AND le.valuenum IS NOT NULL
      AND le.valuenum  > 0
    LIMIT {limit}
    """
    raw = run_query(sql, tuple(hadm_ids) + tuple(all_item_ids))
    if raw.empty:
        log.warning("No labevents found.")
        return pd.DataFrame()

    id_to_name = {}
    for name, ids in LAB_ITEM_IDS.items():
        for i in ids:
            id_to_name[i] = name
    raw["lab_name"] = raw["itemid"].map(id_to_name)

    pivoted = (
        raw.groupby(["hadm_id", "lab_name"])["valuenum"]
        .median()
        .unstack("lab_name")
        .reset_index()
    )
    log.info(f"Labs aggregated: {len(pivoted)} admissions × {len(pivoted.columns)-1} labs")
    return pivoted


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clinical scores and binary flag columns from raw vitals + labs.
    All operations are null-safe.
    """
    out = df.copy()

    # ── Demographics ──────────────────────────────────────────
    if "gender" in out.columns:
        out["gender_M"] = (out["gender"].str.upper() == "M").astype(float)

    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce")
        # MIMIC encodes very old patients as 300 years → clip
        out["age"] = out["age"].clip(upper=120)

    if "admission_type" in out.columns:
        out["is_emergency"] = (
            out["admission_type"].str.upper().isin(["EMERGENCY", "URGENT"])
        ).astype(float)

    # ── MAP ───────────────────────────────────────────────────
    if {"systolic_bp", "diastolic_bp"}.issubset(out.columns):
        out["map"] = out["diastolic_bp"] + (
            (out["systolic_bp"] - out["diastolic_bp"]) / 3
        )

    # ── Vital abnormalities ──────────────────────────────────
    if "heart_rate" in out.columns:
        out["hr_abnormal"] = (
            (out["heart_rate"] < 60) | (out["heart_rate"] > 100)
        ).astype(float)

    if "temperature" in out.columns:
        out["temp_abnormal"] = (
            (out["temperature"] < 36.0) | (out["temperature"] > 38.3)
        ).astype(float)

    if "respiratory_rate" in out.columns:
        out["rr_abnormal"] = (
            (out["respiratory_rate"] < 12) | (out["respiratory_rate"] > 20)
        ).astype(float)

    if "spo2" in out.columns:
        out["hypoxia"] = (out["spo2"] < 94).astype(float)

    if "map" in out.columns:
        out["hypotension"] = (out["map"] < 65).astype(float)

    # ── Lab abnormalities ────────────────────────────────────
    if "wbc" in out.columns:
        out["wbc_abnormal"] = (
            (out["wbc"] < 4) | (out["wbc"] > 12)
        ).astype(float)

    if "lactate" in out.columns:
        out["lactate_high"]   = (out["lactate"] > 2.0).astype(float)
        out["lactate_severe"] = (out["lactate"] > 4.0).astype(float)

    if "creatinine" in out.columns:
        out["aki"] = (out["creatinine"] > 1.5).astype(float)

    if "platelets" in out.columns:
        out["thrombocytopenia"] = (out["platelets"] < 150).astype(float)

    # ── SIRS score (0-4) ─────────────────────────────────────
    sirs = pd.Series(0.0, index=out.index)
    if "temperature" in out.columns:
        sirs += ((out["temperature"] > 38.0) | (out["temperature"] < 36.0)).astype(float)
    if "heart_rate" in out.columns:
        sirs += (out["heart_rate"] > 90).astype(float)
    if "respiratory_rate" in out.columns:
        sirs += (out["respiratory_rate"] > 20).astype(float)
    if "wbc" in out.columns:
        sirs += ((out["wbc"] > 12) | (out["wbc"] < 4)).astype(float)
    out["sirs_score"] = sirs
    out["meets_sirs"]  = (sirs >= 2).astype(float)

    # ── qSOFA score (0-3) ────────────────────────────────────
    qsofa = pd.Series(0.0, index=out.index)
    if "respiratory_rate" in out.columns:
        qsofa += (out["respiratory_rate"] >= 22).astype(float)
    if "systolic_bp" in out.columns:
        qsofa += (out["systolic_bp"] <= 100).astype(float)
    # GCS not easily available without more joins → omit
    out["qsofa_score"] = qsofa
    out["high_qsofa"]  = (qsofa >= 2).astype(float)

    # ── SOFA sub-scores ──────────────────────────────────────
    if "platelets" in out.columns:
        out["sofa_coag"] = pd.cut(
            out["platelets"],
            bins=[-np.inf, 20, 50, 100, 150, np.inf],
            labels=[4, 3, 2, 1, 0]
        ).astype(float)

    if "creatinine" in out.columns:
        out["sofa_renal"] = pd.cut(
            out["creatinine"],
            bins=[-np.inf, 1.2, 1.9, 3.4, 4.9, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    if "bilirubin" in out.columns:
        out["sofa_liver"] = pd.cut(
            out["bilirubin"],
            bins=[-np.inf, 1.2, 1.9, 5.9, 11.9, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    # ── Age groups ───────────────────────────────────────────
    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"],
            bins=[0, 30, 50, 65, 80, 150],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    # ── Composite severity flag ───────────────────────────────
    severity_cols = [c for c in ["lactate_severe", "aki", "thrombocytopenia",
                                  "hypotension", "hypoxia"] if c in out.columns]
    if severity_cols:
        out["organ_dysfunction"] = (out[severity_cols].sum(axis=1) >= 2).astype(float)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_COLS = {
    "subject_id", "hadm_id", "icustay_id", "has_sepsis",
    "gender", "admission_type", "ethnicity",
    "hospital_expire_flag", "dob", "admittime", "dischtime",
}

def prepare_Xy(df: pd.DataFrame):
    """Drop non-feature columns, handle types, impute, return X, y, feature_names."""
    y = df["has_sepsis"].astype(int)

    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")

    # Encode any remaining categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = pd.factorize(X[col])[0].astype(float)

    # Ensure float64 throughout
    X = X.astype(float)

    feature_names = list(X.columns)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_arr   = imputer.fit_transform(X)
    X       = pd.DataFrame(X_arr, columns=feature_names, index=X.index)

    log.info(f"Feature matrix: {X.shape}, Positive rate: {y.mean()*100:.1f}%")
    return X, y, feature_names, imputer


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        log.warning("XGBoost not installed – skipping")
        xgb = None

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
            n_jobs=-1,
        ),
    }
    if xgb is not None:
        models["xgboost"] = xgb
    return models


def train_all_models(X_train, X_test, y_train, y_test):
    models = build_models()
    results = {}
    best_f1, best_name, best_model_obj = 0.0, None, None

    for name, model in models.items():
        log.info(f"  Training {name} …")
        try:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy":  accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall":    recall_score(y_test, y_pred, zero_division=0),
                "f1_score":  f1_score(y_test, y_pred, zero_division=0),
                "roc_auc":   roc_auc_score(y_test, y_proba),
            }
            results[name] = {"model": model, "metrics": metrics}

            log.info(
                f"    Accuracy={metrics['accuracy']:.3f}  "
                f"F1={metrics['f1_score']:.3f}  "
                f"AUC={metrics['roc_auc']:.3f}"
            )

            if metrics["f1_score"] > best_f1:
                best_f1        = metrics["f1_score"]
                best_name      = name
                best_model_obj = model

        except Exception as exc:
            log.error(f"  {name} failed: {exc}")

    log.info(f"\nBest model → {best_name}  (F1={best_f1:.4f})")
    return results, best_name, best_model_obj


# ─────────────────────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────────────────────────────────────

def build_shap_explainer(model, X_train_sample: pd.DataFrame, model_name: str):
    """Build a SHAP explainer appropriate for the model type."""
    try:
        n = min(200, len(X_train_sample))
        background = X_train_sample.sample(n, random_state=42)

        tree_models = ("random_forest", "gradient_boosting", "xgboost")
        if any(t in model_name for t in tree_models):
            explainer = shap.TreeExplainer(model, background)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, background, link="logit"
            )
        log.info("SHAP explainer built successfully")
        return explainer
    except Exception as exc:
        log.error(f"SHAP explainer failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK SYNTHETIC DATA (used when DB has no sepsis labels or is empty)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 5000) -> pd.DataFrame:
    """
    Generates clinically realistic synthetic data so the model can still be
    trained and the app can run while you troubleshoot the MIMIC connection.
    """
    log.warning(
        "⚠  Falling back to SYNTHETIC data.  "
        "Train on real MIMIC data for a proper model."
    )
    rng = np.random.default_rng(42)

    # Positive class (30%)
    n_pos = int(n * 0.30)
    n_neg = n - n_pos

    def block(n_rows, sepsis):
        hr_mu   = 110 if sepsis else 80
        temp_mu = 38.8 if sepsis else 37.0
        rr_mu   = 24  if sepsis else 16
        sbp_mu  = 95  if sepsis else 120
        wbc_mu  = 14  if sepsis else 8
        lac_mu  = 3.5 if sepsis else 1.0
        cre_mu  = 1.8 if sepsis else 0.9
        plt_mu  = 120 if sepsis else 250

        return {
            "age":              rng.integers(20, 85, n_rows).astype(float),
            "gender_M":         rng.choice([0, 1], n_rows).astype(float),
            "is_emergency":     rng.choice([0, 1], n_rows, p=[0.3, 0.7] if sepsis else [0.6, 0.4]).astype(float),
            "heart_rate":       rng.normal(hr_mu,   15, n_rows).clip(40, 180),
            "temperature":      rng.normal(temp_mu,  0.8, n_rows).clip(35, 42),
            "respiratory_rate": rng.normal(rr_mu,    5, n_rows).clip(8, 45),
            "systolic_bp":      rng.normal(sbp_mu,  20, n_rows).clip(70, 200),
            "diastolic_bp":     rng.normal(sbp_mu * 0.65, 12, n_rows).clip(40, 130),
            "spo2":             rng.normal(93 if sepsis else 97, 3, n_rows).clip(80, 100),
            "wbc":              rng.normal(wbc_mu,  4, n_rows).clip(1, 40),
            "lactate":          rng.exponential(lac_mu, n_rows).clip(0.3, 15),
            "creatinine":       rng.exponential(cre_mu, n_rows).clip(0.3, 10),
            "platelets":        rng.normal(plt_mu, 80, n_rows).clip(20, 600),
            "bilirubin":        rng.exponential(1.0 if sepsis else 0.5, n_rows).clip(0.1, 20),
            "glucose":          rng.normal(130 if sepsis else 100, 30, n_rows).clip(50, 400),
            "has_sepsis":       [1 if sepsis else 0] * n_rows,
        }

    pos_df = pd.DataFrame(block(n_pos, sepsis=True))
    neg_df = pd.DataFrame(block(n_neg, sepsis=False))
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save_model():
    print("\n" + "="*65)
    print("  SEPSIS PREDICTION MODEL TRAINING")
    print("  Project: Explainable Multimodal AI-based Sepsis System")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*65 + "\n")

    # ── 1. Load from MIMIC-III ───────────────────────────────
    log.info("STEP 1 ▸ Loading data from MIMIC-III …")
    use_real_data = False

    try:
        admissions = load_admissions(limit=20000)
        if admissions.empty:
            raise ValueError("No admissions returned")

        hadm_ids = admissions["hadm_id"].dropna().astype(int).tolist()
        labels   = load_sepsis_labels(hadm_ids)

        vitals = load_vitals(hadm_ids, limit=200000)
        labs   = load_labs(hadm_ids, limit=200000)

        # Merge
        df = admissions.merge(labels, on="hadm_id", how="left")
        df["has_sepsis"] = df["has_sepsis"].fillna(0).astype(int)

        if not vitals.empty:
            df = df.merge(vitals, on="hadm_id", how="left")
        if not labs.empty:
            df = df.merge(labs, on="hadm_id", how="left")

        # Need at least some positive cases
        if df["has_sepsis"].sum() < 10:
            raise ValueError(
                f"Only {df['has_sepsis'].sum()} sepsis cases found – "
                "check ICD-9 codes or diagnosis text"
            )

        use_real_data = True
        log.info(f"Real MIMIC data ready: {len(df)} admissions")

    except Exception as exc:
        log.error(f"MIMIC data loading failed: {exc}")
        log.warning("Switching to synthetic fallback data …")
        df = generate_synthetic_data(n=8000)

    # ── 2. Feature Engineering ───────────────────────────────
    log.info("STEP 2 ▸ Engineering features …")
    df_eng = engineer_features(df)

    X, y, feature_names, imputer = prepare_Xy(df_eng)

    # ── 3. Train / Test split ────────────────────────────────
    log.info("STEP 3 ▸ Splitting data (80/20 stratified) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 4. SMOTE oversampling (on training set only) ─────────
    pos_rate = y_train.mean()
    if pos_rate < 0.35:
        log.info(
            f"STEP 3b ▸ Applying SMOTE (positive rate = {pos_rate:.1%}) …"
        )
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
            log.info(f"  After SMOTE: {len(X_train)} samples, "
                     f"{y_train.mean()*100:.1f}% positive")
        except Exception as exc:
            log.warning(f"SMOTE failed ({exc}), continuing without it")

    # ── 5. Train models ──────────────────────────────────────
    log.info("STEP 4 ▸ Training models …")
    results, best_name, best_model = train_all_models(
        X_train, X_test, y_train, y_test
    )

    # ── 6. Print full classification report ─────────────────
    log.info("\nClassification Report (best model on test set):")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["No Sepsis", "Sepsis"]))

    # ── 7. Save artifacts ────────────────────────────────────
    log.info("STEP 5 ▸ Saving model artifacts …")

    # 7a – Main model
    model_path = os.path.join(SAVE_DIR, "sepsis_model.pkl")
    joblib.dump({
        "model":            best_model,
        "model_name":       best_name,
        "feature_importance": (
            pd.DataFrame({
                "feature":    feature_names,
                "importance": (
                    best_model.feature_importances_
                    if hasattr(best_model, "feature_importances_")
                    else np.abs(best_model.coef_[0])
                    if hasattr(best_model, "coef_")
                    else np.zeros(len(feature_names))
                )
            }).sort_values("importance", ascending=False)
        ),
        "all_results":      {k: v["metrics"] for k, v in results.items()},
        "trained_on_real":  use_real_data,
        "trained_at":       datetime.now().isoformat(),
    }, model_path)
    log.info(f"  Model saved  → {model_path}")

    # 7b – Feature names
    fn_path = os.path.join(SAVE_DIR, "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"  Features     → {fn_path}")

    # 7c – Imputer + scaler bundle (called feature_engineer.pkl for compatibility)
    scaler  = StandardScaler()
    scaler.fit(X_train)
    fe_path = os.path.join(SAVE_DIR, "feature_engineer.pkl")
    with open(fe_path, "wb") as f:
        pickle.dump({"imputer": imputer, "scaler": scaler}, f)
    log.info(f"  Preprocessor → {fe_path}")

    # 7d – SHAP explainer
    log.info("STEP 6 ▸ Building SHAP explainer …")
    shap_explainer = build_shap_explainer(best_model, X_train, best_name)
    if shap_explainer is not None:
        shap_path = os.path.join(SAVE_DIR, "shap_explainer.pkl")
        with open(shap_path, "wb") as f:
            pickle.dump(shap_explainer, f)
        log.info(f"  SHAP         → {shap_path}")
    else:
        log.warning("SHAP explainer not saved (see error above)")

    # ── 8. Summary ──────────────────────────────────────────
    best_metrics = results[best_name]["metrics"]
    print("\n" + "="*65)
    print("  TRAINING COMPLETE")
    print(f"  Best model : {best_name}")
    print(f"  Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"  Precision  : {best_metrics['precision']:.4f}")
    print(f"  Recall     : {best_metrics['recall']:.4f}")
    print(f"  F1-Score   : {best_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC    : {best_metrics['roc_auc']:.4f}")
    print(f"  Data source: {'MIMIC-III (real)' if use_real_data else 'SYNTHETIC'}")
    print("="*65)
    print("\nNext steps:")
    print("  1.  python run.py")
    print("  2.  Open http://localhost:5000")
    print("  3.  Login → admin / admin123")
    print()


if __name__ == "__main__":
    train_and_save_model()