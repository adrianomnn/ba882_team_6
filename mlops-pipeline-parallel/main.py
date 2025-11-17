"""
ML Training Cloud Function - Async version that returns immediately
Runs training in background, logs progress to BigQuery
"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import json, joblib, tempfile, os
from datetime import datetime
from google.cloud import bigquery, storage
import functions_framework
import pandas as pd
import numpy as np

bq_client = bigquery.Client()
storage_client = storage.Client()

# Config
PROJECT_ID = os.getenv("GCP_PROJECT", "adrineto-qst882-fall25")
BQ_DATASET = os.getenv("GOLDEN_DATASET", "youtube_golden")
BQ_TABLE = os.getenv("GOLDEN_TABLE", "video_features_for_ml_v3_consecutive")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "adrineto-ba882-fall25-team-6")
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "models/youtube")
LABEL_COL = os.getenv("LABEL_COL", "is_trending_tomorrow")
RANDOM_STATE = 42


# ---------- Helpers ----------
def fetch_golden_features(date: str = None, limit: int = None) -> pd.DataFrame:
    """Fetch training data from BigQuery. If date provided, filter snapshot_date = date."""
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    if date:
        # Expecting date in "YYYY-MM-DD" format
        sql = f"SELECT * FROM {table_ref} WHERE snapshot_date = @date"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("date", "DATE", date)]
        )
        df = bq_client.query(sql, job_config=job_config).to_dataframe()
    else:
        sql = f"SELECT * FROM {table_ref}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        df = bq_client.query(sql).to_dataframe()
    return df

def select_features_and_label(df: pd.DataFrame):
    """Prepare features and labels"""
    candidate_features = [
        "views", "views_lag_1d", "view_delta_1d", "view_delta_pct_1d", "view_accel_2d",
        "likes", "like_delta_1d", "like_rate_t", "engagement_rate_t", "engagement_growth",
        "days_since_publish", "age_bucket", "upload_hour", "upload_weekday", "pr_view_accel_pos"
    ]
    features = [c for c in candidate_features if c in df.columns]
    label = LABEL_COL
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in golden table")

    X = df[features].copy().fillna(0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    return X, y, features

def select_model(algorithm: str, hyperparams: dict):
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **(hyperparams or {}))
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **(hyperparams or {}))
    if algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **(hyperparams or {}))
    raise ValueError(f"Unsupported algorithm: {algorithm}")

def compute_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_proba = None

    # safe PR-AUC and ROC-AUC only if both classes are present in y_test
    has_both_classes = len(np.unique(y_test)) > 1
    pr_auc = None
    roc_auc = None
    try:
        if y_proba is not None and has_both_classes:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = float(auc(recall_curve, precision_curve))
        if y_proba is not None and has_both_classes:
            roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        # leave as None if fails
        pr_auc = pr_auc
        roc_auc = roc_auc

    # confusion matrix safe
    try:
        cm = confusion_matrix(y_test, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel().tolist()
        else:
            tn = fp = fn = tp = None
    except Exception:
        tn = fp = fn = tp = None

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positive": int(tp) if tp is not None else None,
        "false_positive": int(fp) if fp is not None else None,
        "false_negative": int(fn) if fn is not None else None,
        "true_negative": int(tn) if tn is not None else None
    }

def save_model_to_gcs(model, bucket_name, object_path):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_path)
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        joblib.dump(model, tmp.name)
        tmp.flush()
        blob.upload_from_filename(tmp.name)
        os.unlink(tmp.name)
    gcs_path = f"gs://{bucket_name}/{object_path}"
    print(f"Model saved to {gcs_path}")
    return gcs_path

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
        bigquery.SchemaField("train_size", "INTEGER"),
        bigquery.SchemaField("test_size", "INTEGER"),
    ]
    try:
        bq_client.get_table(table_id)
    except Exception:
        table_def = bigquery.Table(table_id, schema=schema)
        bq_client.create_table(table_def)
    rows = [{
        "run_id": metadata["run_id"],
        "created_at": metadata["created_at"],
        "model_gcs_path": metadata["model_gcs_path"],
        "algorithm": metadata["algorithm"],
        "hyperparams": json.dumps(metadata.get("hyperparams", {})),
        "metrics": json.dumps(metadata.get("metrics", {})),
        "num_rows": metadata.get("num_rows", 0),
        "features": json.dumps(metadata.get("features", [])),
        "train_size": metadata.get("train_size", 0),
        "test_size": metadata.get("test_size", 0),
    }]
    bq_client.insert_rows_json(table_id, rows)
    print(f"Metadata written to {table_id} for run_id={metadata['run_id']}")

# ---------- Entrypoint ----------
@functions_framework.http
def train_model(request):
    start_time = datetime.utcnow()
    try:
        payload = request.get_json(silent=True) or {}
        algorithm = payload.get("algorithm", "random_forest")
        hyperparams = payload.get("hyperparameters", {}) or {}
        limit = payload.get("limit")
        snapshot_date = payload.get("snapshot_date")  # expect "YYYY-MM-DD" or None
        run_id = payload.get("run_id") or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")

        print(f"Starting training: algorithm={algorithm}, run_id={run_id}, snapshot_date={snapshot_date}")

        # 1. Fetch data (respecting snapshot_date if provided)
        df = fetch_golden_features(date=snapshot_date, limit=limit)

        if df is None or df.shape[0] == 0:
            return jsonify({"status": "failed", "error": "No data found for training", "run_id": run_id}), 400

        # If snapshot_date column exists, coerce to datetime for correct quantile logic
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
            df = df.dropna(subset=["snapshot_date"])  # drop rows where snapshot_date could not parse

        # Sort by snapshot_date if present
        if "snapshot_date" in df.columns:
            df = df.sort_values("snapshot_date")

        # 2. Prepare features + label
        X, y, feature_cols = select_features_and_label(df)

        # 3. Time-aware train/test split using quantile cutoff
        if "snapshot_date" in df.columns and df["snapshot_date"].nunique() > 1:
            cutoff_ts = df["snapshot_date"].quantile(0.7)
            train_mask = df["snapshot_date"] <= cutoff_ts
            X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
            y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
        else:
            # fallback to standard split when no timestamp variety
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
            )

        # Validate training/test sizes
        if len(X_train) < 5 or len(X_test) < 5:
            msg = f"Insufficient train/test size: train={len(X_train)}, test={len(X_test)}"
            print(msg)
            return jsonify({"status": "failed", "error": msg, "run_id": run_id}), 400

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        # 4. Train
        model = select_model(algorithm, hyperparams)
        model.fit(X_train, y_train)

        # 5. Evaluate (safe)
        metrics = compute_metrics(model, X_test, y_test)

        # 6. Persist model
        gcs_rel_path = f"{MODEL_PREFIX}/{run_id}/{algorithm}/model.pkl"
        gcs_full_path = save_model_to_gcs(model, MODEL_BUCKET, gcs_rel_path)

        # 7. Save metadata to BQ
        metadata = {
            "run_id": run_id,
            "created_at": start_time.isoformat(),
            "model_gcs_path": gcs_full_path,
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "num_rows": len(df),
            "features": feature_cols,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        write_training_metadata_to_bq(metadata)

        duration = (datetime.utcnow() - start_time).total_seconds()
        result = {
            "status": "success",
            "run_id": run_id,
            "algorithm": algorithm,
            "metrics": metrics,
            "model_path": gcs_full_path,
            "duration_seconds": duration,
            "num_rows": len(df)
        }
        print(f"Training complete for run_id={run_id} in {duration:.1f}s")
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "failed", "error": str(e), "run_id": locals().get("run_id", None)}), 500