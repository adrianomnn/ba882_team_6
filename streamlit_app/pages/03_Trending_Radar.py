import streamlit as st
from utils.bq import run_query

st.title("Trending Radar")

PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]

with st.sidebar:
    min_prob = st.slider("Min trending probability", 0.50, 0.99, 0.80, 0.01)
    model = st.text_input("Filter by model version (optional)")

where = "WHERE p.y_prob >= CAST(@minp AS FLOAT64)"
params = {"minp": str(min_prob)}
if model:
    where += " AND p.model_version = @m"
    params["m"] = model

sql = f"""
SELECT
  p.snapshot_date, p.model_version,
  p.video_id, p.channel_id, p.published_at,
  p.y_prob, p.y_pred,
  v.title, c.channel_title
FROM `ml_artifacts.v_latest_predictions` p
LEFT JOIN `youtube_staging.dim_videos` v ON v.video_id = p.video_id
LEFT JOIN `youtube_staging.dim_channels` c ON c.channel_id = p.channel_id
{where}
ORDER BY p.y_prob DESC
LIMIT 200
"""
df = run_query(sql, params=params)

st.caption("Shows the most recent scoring run from ml_artifacts.predictions_daily")
if not df.empty:
    st.dataframe(df)
    st.download_button("Download CSV", df.to_csv(index=False).encode(), "trending_candidates.csv","text/csv")
else:
    st.info("No candidates match the filters yet.")

