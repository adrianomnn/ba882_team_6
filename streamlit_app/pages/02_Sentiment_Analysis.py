import streamlit as st
import pandas as pd
import numpy as np
from utils.bq import run_query
from google.cloud import storage
from google.oauth2 import service_account

# -------------------------------------------
# CONFIG
# -------------------------------------------
PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET  = st.secrets.get("bq_dataset", "youtube_staging")

# ---------------- App Page Title -----------------
st.set_page_config(page_title="Weekley Comments Sentiment Analysis", layout="wide")
st.title("Weekley Comments Sentiment Analysis")

# ---------------- Comments Table ----------------
comments_sql = f"""
WITH f_comments AS (
  SELECT
    comment_id,
    published_at
  FROM `{PROJECT}.{DATASET}.fact_comments`
),
d_comments AS (
  SELECT 
    comment_id, 
    comment_text
  FROM `{PROJECT}.{DATASET}.dim_comments`
)
SELECT *
FROM f_comments
JOIN d_comments 
  ON f_comments.comment_id = d_comments.comment_id
"""
df = run_query(comments_sql)

st.dataframe(df.head(20))
