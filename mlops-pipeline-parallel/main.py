from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import json, joblib, io, os, tempfile
from datetime import datetime
from google.cloud import bigquery, storage
from flask import Request, jsonify
import pandas as pd
import numpy as np

bq_client = bigquery.Client()
storage_client = storage.Client()

# Config via environment variables
PROJECT_ID = os.getenv("GCP_PROJECT", "adrineto-qst882-fall25")
BQ_DATASET = os.getenv("GOLDEN_DATASET", "youtube_golden")
BQ_TABLE = os.getenv("GOLDEN_TABLE", "video_features_for_ml_v3_consecutive")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "adrineto-ba882-fall25-team-6")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models_parallel/youtube")
LABEL_COL = os.getenv("LABEL_COL", "is_trending_tomorrow")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# --- 1. Fetch Data ---
def fetch_golden_features(date: str = None, limit: int = None) -> pd.DataFrame:
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    if date:
        sql = f"SELECT * FROM {table_ref} WHERE snapshot_date = @date"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("date", "DATE", date)]
        )
        job = bq_client.query(sql, job_config=job_config)
    else:
        sql = f"SELECT * FROM {table_ref}"
        job = bq_client.query(sql)

    df = job.to_dataframe()
    if limit:
        df = df.head(limit)
    return df


# --- 2. Prepare Features & Label ---
def select_features_and_label(df: pd.DataFrame):
    candidate_features = [
        "views",
        "views_lag_1d",
        "view_delta_1d",
        "view_delta_pct_1d",
        "view_accel_2d",
        "likes",
        "like_delta_1d",
        "like_rate_t",
        "engagement_rate_t",
        "engagement_growth",
        "days_since_publish",
        "age_bucket",
        "upload_hour",
        "upload_weekday"
        "pr_view_accel_pos"
    ]

    features = [c for c in candidate_features if c in df.columns]
    label = LABEL_COL
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in golden table")

    X = df[features].fillna(0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    return X, y, features


# --- 3. Model Selection ---
def select_model(algorithm: str, hyperparams: dict):
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **hyperparams)
    elif algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **hyperparams)
    elif algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **hyperparams)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


# --- 4. Compute Metrics ---
def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Probabilities only if model supports predict_proba
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # PR-AUC (only if probabilities exist)
    if y_proba is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist() if cm.size == 4 else (None, None, None, None)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "pr_auc": pr_auc,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn
    }


# --- 5. Save Model to GCS ---
def save_model_to_gcs(model, bucket_name, object_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        joblib.dump(model, tmp.name)
        tmp.flush()
        blob.upload_from_filename(tmp.name)
    return f"gs://{bucket_name}/{object_path}"


# --- 6. Write Metadata to BigQuery ---
def write_training_metadata_to_bq(metadata: dict):
    dataset = os.getenv("TRAINING_METADATA_DATASET", "youtube_metadata")
    table = os.getenv("TRAINING_METADATA_TABLE", "training_runs_parallel")
    table_id = f"{PROJECT_ID}.{dataset}.{table}"

    schema = [
        bigquery.SchemaField("run_id", "STRING"),
        bigquery.SchemaField("created_at", "TIMESTAMP"),
        bigquery.SchemaField("model_gcs_path", "STRING"),
        bigquery.SchemaField("algorithm", "STRING"),
        bigquery.SchemaField("hyperparams", "STRING"),
        bigquery.SchemaField("metrics", "STRING"),
        bigquery.SchemaField("num_rows", "INTEGER"),
        bigquery.SchemaField("features", "STRING"),
    ]

    # Create table if not exists
    try:
        bq_client.get_table(table_id)
    except Exception:
        table_def = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table_def)
        print(f"Created table {table_id}")

    rows = [{
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get("created_at"),
        "model_gcs_path": metadata.get("model_gcs_path"),
        "algorithm": metadata.get("algorithm"),
        "hyperparams": json.dumps(metadata.get("hyperparams", {})),
        "metrics": json.dumps(metadata.get("metrics", {})),
        "num_rows": metadata.get("num_rows", 0),
        "features": json.dumps(metadata.get("features", [])),
    }]
    bq_client.insert_rows_json(table_id, rows)
    print(f"Metadata written to {table_id} for run_id={metadata['run_id']}")


# --- 7. HTTP Entrypoint ---
def http_entry(request: Request):
    try:
        payload = {}
        if request.method == "POST":
            payload = request.get_json(silent=True) or {}

        algorithm = payload.get("algorithm", "random_forest")
        hyperparams = payload.get("hyperparameters", {})
        snapshot_date = payload.get("snapshot_date")
        limit = payload.get("limit")
        run_id = payload.get("run_id") or datetime.utcnow().strftime("run_%Y%m%dT%H%M%S")

        df = fetch_golden_features(date=None, limit=limit)
        # Ensure data sorted by snapshot_date
        df = df.sort_values("snapshot_date")
        X, y, feature_cols = select_features_and_label(df)

        cutoff = df["snapshot_date"].quantile(0.7)
        train_idx = df["snapshot_date"] <= cutoff
        test_idx  = df["snapshot_date"] > cutoff 

        if train_idx.sum() == 0:
            raise ValueError("Training set is empty — check snapshot_date values.")

        if test_idx.sum() == 0:
            raise ValueError("Test set is empty — cannot compute metrics.")

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]

        model = select_model(algorithm, hyperparams)
        model.fit(X_train, y_train)
        metrics = compute_metrics(model, X_test, y_test)

        gcs_path = f"{MODEL_PREFIX}/{run_id}/{algorithm}/model.pkl"
        full_gcs_path = save_model_to_gcs(model, MODEL_BUCKET, gcs_path)

        metadata = {
            "run_id": run_id,
            "created_at": datetime.utcnow().isoformat(),
            "model_gcs_path": full_gcs_path,
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "num_rows": len(df),
            "features": feature_cols,
        }
        write_training_metadata_to_bq(metadata)

        return jsonify({"status": "success", "metadata": metadata}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "error", "message": str(e)}), 500
