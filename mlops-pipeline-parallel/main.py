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


def fetch_golden_features(date: str = None, limit: int = None) -> pd.DataFrame:
    """Fetch training data from BigQuery"""
    table_ref = f"`{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"
    
    if date:
        sql = f"SELECT * FROM {table_ref} WHERE snapshot_date = @date"
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("date", "DATE", date)]
        )
        df = bq_client.query(sql, job_config=job_config).to_dataframe()
    else:
        sql = f"SELECT * FROM {table_ref}"
        if limit:
            sql += f" LIMIT {limit}"
        df = bq_client.query(sql).to_dataframe()
    
    return df


def select_features_and_label(df: pd.DataFrame):
    """Prepare features and labels"""
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
        "upload_weekday",
        "pr_view_accel_pos"
    ]

    features = [c for c in candidate_features if c in df.columns]
    label = LABEL_COL
    
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found")

    X = df[features].fillna(0)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    y = pd.to_numeric(df[label], errors="coerce").fillna(0).astype(int)
    
    return X, y, features


def select_model(algorithm: str, hyperparams: dict):
    """Initialize model based on algorithm"""
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, **hyperparams)
    elif algorithm == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE, **hyperparams)
    elif algorithm == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, **hyperparams)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def compute_metrics(model, X_test, y_test):
    """Compute all evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # PR-AUC
    if y_proba is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
    else:
        pr_auc = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel().tolist() if cm.size == 4 else (None, None, None, None)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "pr_auc": float(pr_auc) if pr_auc else None,
        "true_positive": int(tp) if tp is not None else None,
        "false_positive": int(fp) if fp is not None else None,
        "false_negative": int(fn) if fn is not None else None,
        "true_negative": int(tn) if tn is not None else None
    }


def save_model_to_gcs(model, bucket_name, object_path):
    """Save trained model to GCS"""
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
    """Write training run metadata to BigQuery"""
    dataset = "youtube_metadata"
    table = "training_runs_parallel"
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

    # Create table if not exists
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


@functions_framework.http
def train_model(request):
    """
    HTTP entry point - Trains model synchronously
    Returns 200 when complete
    """
    start_time = datetime.utcnow()
    
    try:
        payload = request.get_json(silent=True) or {}
        
        # Parse parameters
        algorithm = payload.get("algorithm", "random_forest")
        hyperparams = payload.get("hyperparameters", {})
        limit = payload.get("limit")
        snapshot_date = payload.get("snapshot_date")
        run_id = payload.get("run_id") or datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        
        print(f"Starting training: {algorithm} | run_id={run_id}")
        
        # 1. Fetch data
        df = fetch_golden_features(date=None, limit=None)
        if len(df) == 0:
            return {"error": "No data found", "status": "failed"}, 400
        
        # Sort by date for time-aware split
        if "snapshot_date" in df.columns:
            df = df.sort_values("snapshot_date")
        
        # 2. Prepare features
        X, y, feature_cols = select_features_and_label(df)
        
        # 3. Time-aware train/test split
        if "snapshot_date" in df.columns:
            cutoff = df["snapshot_date"].quantile(0.7)
            train_mask = df["snapshot_date"] <= cutoff
            X_train, X_test = X[train_mask], X[~train_mask]
            y_train, y_test = y[train_mask], y[~train_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
        
        print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
        
        # 4. Train model
        print(f"Training {algorithm}...")
        model = select_model(algorithm, hyperparams)
        model.fit(X_train, y_train)
        print(f"Training complete")
        
        # 5. Evaluate
        print("Computing metrics...")
        metrics = compute_metrics(model, X_test, y_test)
        
        # 6. Save model
        gcs_path = f"{MODEL_PREFIX}/{run_id}/{algorithm}/model.pkl"
        full_path = save_model_to_gcs(model, MODEL_BUCKET, gcs_path)
        
        # 7. Save metadata
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        metadata = {
            "run_id": run_id,
            "created_at": start_time.isoformat(),
            "model_gcs_path": full_path,
            "algorithm": algorithm,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "num_rows": len(df),
            "features": feature_cols,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        
        write_training_metadata_to_bq(metadata)
        
        result = {
            "status": "success",
            "run_id": run_id,
            "algorithm": algorithm,
            "metrics": metrics,
            "model_path": full_path,
            "duration_seconds": duration,
            "num_rows": len(df)
        }
        
        print(f"Training pipeline complete in {duration:.1f}s")
        print(f"   Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f}")
        
        return result, 200
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        print(f"{error_msg}")
        import traceback
        traceback.print_exc()
        
        return {
            "status": "failed",
            "error": str(e),
            "run_id": locals().get("run_id", "unknown")
        }, 500