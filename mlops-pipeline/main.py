# main.py
import os
import json
import tempfile
from datetime import datetime
from flask import jsonify, Request

import pandas as pd
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

# Config via environment variables
PROJECT_ID = os.getenv("GCP_PROJECT", "adrineto-qst882-fall25")
BQ_DATASET = os.getenv("GOLDEN_DATASET", "youtube_golden")
BQ_TABLE = os.getenv("GOLDEN_TABLE", "video_features_for_ml_v3_consecutive")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "adrineto-ba882-fall25-team-6")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/youtube")
MODEL_NAME = os.getenv("MODEL_NAME", "rf_youtube_trending.joblib")
LABEL_COL = os.getenv("LABEL_COL", "is_trending")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)


def fetch_golden_features(date: str = None, limit: int = None) -> pd.DataFrame:
    """
    Read table from BigQuery. Optionally filter by snapshot_date equal to `date`.
    """
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    if date:
        sql = f"SELECT * FROM {table_ref} WHERE snapshot_date = @date"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("date", "DATE", date),
            ]
        )
        job = bq_client.query(sql, job_config=job_config)
    else:
        sql = f"SELECT * FROM {table_ref}"
        job = bq_client.query(sql)

    df = job.to_dataframe()
    if limit:
        df = df.head(limit)
    return df


def select_features_and_label(df: pd.DataFrame):
    """
    Minimal safe feature selection based on your golden schema.
    - Drop columns that are identifiers for features used only for grouping.
    - Keep numeric features and engineered fields. Convert dtypes safely.
    """

    candidate_features = [
        "views",
        "likes",
        "view_delta_1d",
        "views_lag_1d,",
        "view_accel_2d",
        "engagement_rate_t",
        "engagement_growth",
        "age_bucket"
    ]

    # Keep only columns present in df
    features = [c for c in candidate_features if c in df.columns]
    label = LABEL_COL
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in golden table")

    # Fill NA for numeric features
    X = df[features].copy()
    X = X.fillna(0)
    # Ensure numeric dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    return X, y, features


def train_model(X_train, y_train):
    """
    Train a RandomForest classifier
    """
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def save_model_to_gcs(model, bucket_name, object_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        joblib.dump(model, tmp.name)
        tmp.flush()
        blob.upload_from_filename(tmp.name)
    return f"gs://{bucket_name}/{object_path}"


def compute_metrics(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    return metrics


def write_training_metadata_to_bq(metadata: dict):
    """
    Append a metadata row to a training_log table
      dataset.training_runs : run_id STRING, created_at TIMESTAMP, model_gcs_path STRING, metrics JSON
    """
    dataset = os.getenv("TRAINING_METADATA_DATASET", "youtube_metadata")
    table = os.getenv("TRAINING_METADATA_TABLE", "training_runs")

    table_id = f"{PROJECT_ID}.{dataset}.{table}"
    rows = [
        {
            "run_id": metadata.get("run_id"),
            "created_at": metadata.get("created_at"),
            "model_gcs_path": metadata.get("model_gcs_path"),
            "metrics": json.dumps(metadata.get("metrics", {})),
            "num_rows": metadata.get("num_rows", 0),
            "features": json.dumps(metadata.get("features", [])),
        }
    ]
    bq_client.insert_rows_json(table_id, rows)


def build_run_id():
    return datetime.utcnow().strftime("run_%Y%m%dT%H%M%S")


def http_entry(request: Request):
    """
    Cloud Function HTTP entrypoint.
    Accepts optional JSON payload:
      { "snapshot_date": "2025-10-14", "limit": 10000 }
    Returns JSON with training results and the GCS model path.
    """
    try:
        payload = {}
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}

        snapshot_date = payload.get("snapshot_date")
        limit = payload.get("limit")  # useful for dev/test
        run_id = build_run_id()

        # 1) fetch data
        df = fetch_golden_features(snapshot_date, limit)

        if df.empty or len(df) < 10:
            return jsonify({
                "status": "failed",
                "message": "Not enough data for training",
                "num_rows": len(df)
            }), 400

        # 2) features + label
        X, y, feature_cols = select_features_and_label(df)

        # 3) split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(y.unique()) > 1 else None
        )

        # 4) train
        model = train_model(X_train, y_train)

        # 5) evaluate
        metrics = compute_metrics(model, X_test, y_test)

        # 6) save model to GCS
        model_object_path = f"{MODEL_PREFIX}/{run_id}/{MODEL_NAME}"
        gcs_path = save_model_to_gcs(model, MODEL_BUCKET, model_object_path)

        # 7) write metadata to BQ
        metadata = {
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
            "model_gcs_path": gcs_path,
            "metrics": metrics,
            "num_rows": len(df),
            "features": feature_cols,
        }
        try:
            write_training_metadata_to_bq(metadata)
        except Exception as e:
            # Non-fatal
            print("Warning: could not write training metadata to BigQuery:", e)

        return jsonify({
            "status": "success",
            "run_id": run_id,
            "model_gcs_path": gcs_path,
            "metrics": metrics,
            "num_rows": len(df),
            "features": feature_cols
        }), 200

    except Exception as e:
        print("Error training model:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
