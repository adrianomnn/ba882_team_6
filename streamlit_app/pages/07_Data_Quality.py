import streamlit as st
from utils.bq import run_query

st.title("Data Quality & Freshness")

PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]

fresh_sql = f"""
SELECT 'fact_video_statistics' AS table_name,
       MAX(date) AS latest,
       COUNT(*) AS rows_today
FROM `adrineto-qst882-fall25.youtube_staging.fact_video_statistics`
WHERE date = (SELECT MAX(date) FROM `adrineto-qst882-fall25.youtube_staging.fact_video_statistics`)
UNION ALL
SELECT 'predictions_daily',
       MAX(snapshot_date),
       COUNTIF(snapshot_date = (SELECT MAX(snapshot_date) FROM `adrineto-qst882-fall25.ml_artifacts.predictions_daily`))
FROM `adrineto-qst882-fall25.ml_artifacts.predictions_daily`
UNION ALL
SELECT 'labels_daily',
       MAX(snapshot_date),
       COUNTIF(snapshot_date = (SELECT MAX(snapshot_date) FROM `adrineto-qst882-fall25.ml_artifacts.labels_daily`))
FROM `adrineto-qst882-fall25.ml_artifacts.labels_daily`
UNION ALL
SELECT 'monitoring_daily',
       MAX(snapshot_date),
       COUNTIF(snapshot_date = (SELECT MAX(snapshot_date) FROM `adrineto-qst882-fall25.ml_artifacts.monitoring_daily`))
FROM `adrineto-qst882-fall25.ml_artifacts.monitoring_daily`;
"""
st.subheader("Latest partition dates")
st.dataframe(run_query(fresh_sql))
