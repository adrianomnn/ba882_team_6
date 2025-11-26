import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import json
from google.cloud import storage
from google.oauth2 import service_account
import tempfile
import datetime

# -------------------------------------------
# CONFIG
# -------------------------------------------
GCS_PRODUCTION_BUCKET = "adrineto-ba882-fall25-team-6"
PRODUCTION_MODEL_PATH = "models/youtube/production/model.pkl"
PRODUCTION_METADATA_PATH = "models/youtube/production/metadata.json"

YOUTUBE_API_KEY = st.secrets.get("YOUTUBE_API_KEY") or st.secrets["gcp_service_account"]["YOUTUBE_API_KEY"]


# -------------------------------------------
# HELPERS
# -------------------------------------------

@st.cache_resource
def get_storage_client():
    info = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(info)
    return storage.Client(credentials=credentials, project=info["project_id"])

@st.cache_resource
def load_production_model():
    client = get_storage_client()

    bucket = client.bucket(GCS_PRODUCTION_BUCKET)

    model_blob = bucket.blob(PRODUCTION_MODEL_PATH)
    metadata_blob = bucket.blob(PRODUCTION_METADATA_PATH)

    # download model
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        model_blob.download_to_filename(tmp.name)
        import joblib
        model = joblib.load(tmp.name)

    # download metadata
    metadata = json.loads(metadata_blob.download_as_text())

    return model, metadata


def fetch_youtube_raw(video_id: str):
    """Fetch raw video data from YouTube API."""
    url = (
        f"https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet,statistics,contentDetails"
        f"&id={video_id}&key={YOUTUBE_API_KEY}"
    )
    resp = requests.get(url)
    data = resp.json()

    if "items" not in data or len(data["items"]) == 0:
        return None, "Invalid YouTube video ID"

    return data["items"][0], None

def compute_features(item):
    """Replicate the Golden Table feature set."""

    stats = item.get("statistics", {})
    snippet = item.get("snippet", {})

    views_today = int(stats.get("viewCount", 0))
    likes_today = int(stats.get("likeCount", 0))
    comments_today = int(stats.get("commentCount", 0))

    # --- date fields ---
    published_at = snippet["publishedAt"]
    published_dt = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    today_dt = datetime.datetime.now(datetime.timezone.utc)
    days_since_publish = (today_dt.date() - published_dt.date()).days

    upload_hour = published_dt.hour
    upload_weekday = published_dt.isoweekday()

    # --- age bucket ---
    if days_since_publish <= 1:
        age_bucket = "0-1d"
    elif days_since_publish <= 3:
        age_bucket = "2-3d"
    elif days_since_publish <= 7:
        age_bucket = "4-7d"
    elif days_since_publish <= 30:
        age_bucket = "8-30d"
    else:
        age_bucket = "31d+"

    features = {
        "views": views_today,
        "views_lag_1d": views_today,       
        "view_delta_1d": 0,
        "view_delta_pct_1d": 0,
        "view_accel_2d": 0,
        "likes": likes_today,
        "like_delta_1d": 0,
        "like_rate_t": likes_today / views_today if views_today > 0 else 0,
        "engagement_rate_t": (likes_today + comments_today) / views_today if views_today > 0 else 0,
        "engagement_growth": 0,
        "days_since_publish": days_since_publish,
        "age_bucket": age_bucket,
        "upload_hour": upload_hour,
        "upload_weekday": upload_weekday,
        "pr_view_accel_pos": 0.5,
    }

    return features

def build_feature_dataframe(features: dict):
    """Convert dict into 1-row dataframe with same ordering as training."""
    df = pd.DataFrame([features])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def predict_from_features(model, df):
    """Run model prediction and probability."""
    pred = model.predict(df)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0][1]
    else:
        proba = None

    return pred, proba

# -------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------

st.title("🎬 YouTube Viral Video Predictor")

st.markdown(
    """
    This tool loads the **best production ML model** and predicts whether a YouTube video 
    will go **viral** based only on its video ID.
    """
)

# Load model + metadata (cached)
model, metadata = load_production_model()

# ------------------------------------------
# Display model metadata
# ------------------------------------------
with st.expander("📊 Production Model Details", expanded=True):
    st.json(metadata)

# ------------------------------------------
# Input section
# ------------------------------------------
video_id = st.text_input("Enter a YouTube Video ID:", placeholder="e.g., dQw4w9WgXcQ")

if st.button("Run Prediction"):
    if not video_id:
        st.error("Please enter a YouTube video ID.")
        st.stop()

    # Fetch YouTube data
    with st.spinner("Fetching YouTube data..."):
        item, error = fetch_youtube_raw(video_id)

    if error:
        st.error(error)
        st.stop()

    # Compute today's-only features
    features = compute_features(item)

    st.success("✓ YouTube data fetched successfully")
    st.json(features)

    # Build DF for model
    df = build_feature_dataframe(features)

    # Predict
    pred, proba = predict_from_features(model, df)

    st.subheader("Prediction Result")

    label = "🔥 VIRAL" if pred == 1 else "❄️ NOT VIRAL"

    st.markdown(
        f"""
        ### **{label}**

        **Probability viral:**  
        {proba:.4f}  
        """
    )

    with st.expander("🔍 Model Input Features"):
        st.write(df)

