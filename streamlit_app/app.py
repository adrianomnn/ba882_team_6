import streamlit as st
import pandas as pd
from utils.bq import run_query

st.set_page_config(page_title="YouTube Data Dashboard", layout="wide")
st.title("YouTube Data Dashboard")

PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET  = st.secrets.get("bq_dataset", "youtube_staging")

# ---------------- Sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    ndays_choice = st.radio("Date range", ["Last 7 days", "Last 14 days", "Last 30 days"], index=1)
    ndays = 7 if ndays_choice == "Last 7 days" else 14 if ndays_choice == "Last 14 days" else 30

    channels_sql = f"""
    SELECT DISTINCT c.channel_id, c.channel_title
    FROM `{PROJECT}.{DATASET}.dim_channels` c
    ORDER BY c.channel_title
    """
    ch_df = run_query(channels_sql)
    if ch_df.empty:
        channel = "(All)"
        channel_map = {}
        channel_options = ["(All)"]
    else:
        channel_map = dict(zip(ch_df["channel_title"], ch_df["channel_id"]))
        channel_options = ["(All)"] + list(channel_map.keys())
        channel = st.selectbox("Channel", channel_options, index=0)

# WHERE uses s.date; channel filter via c.channel_id
where = "WHERE s.date >= DATE_SUB(CURRENT_DATE(), INTERVAL @ndays DAY)"
params = {"ndays": str(ndays)}
if channel != "(All)":
    where += " AND c.channel_id = @channel_id"
    params["channel_id"] = channel_map[channel]

# # ---------------- Health ----------------
# st.subheader("Testing BigQuery connection...")
# ok = run_query("SELECT 1 AS ok")
# if ok.empty:
#     st.error("Could not connect to BigQuery.")
#     st.stop()
# else:
#     st.success("✅ Connected to BigQuery successfully!")
#     st.dataframe(ok)

# ---------------- KPIs ----------------
kpis_sql = f"""
SELECT
  SUM(s.view_count)    AS views,
  SUM(s.like_count)    AS likes,
  SUM(s.comment_count) AS comments
FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
JOIN `{PROJECT}.{DATASET}.dim_videos`   v
  ON v.video_id = s.video_id
JOIN `{PROJECT}.{DATASET}.dim_channels` c
  ON c.channel_id = v.channel_id
{where}
"""
kpis = run_query(kpis_sql, params=params)

st.markdown("### KPIs")
c1, c2, c3 = st.columns(3)
if not kpis.empty:
    c1.metric("Total Views",    int(kpis.iloc[0]["views"] or 0))
    c2.metric("Total Likes",    int(kpis.iloc[0]["likes"] or 0))
    c3.metric("Total Comments", int(kpis.iloc[0]["comments"] or 0))
else:
    st.info("No KPI data for this selection.")

# ---------------- Daily time series ----------------
daily_sql = f"""
WITH daily AS (
  SELECT s.date AS d,
         SUM(s.view_count)    AS views,
         SUM(s.like_count)    AS likes,
         SUM(s.comment_count) AS comments
  FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
  JOIN `{PROJECT}.{DATASET}.dim_videos`   v
    ON v.video_id = s.video_id
  JOIN `{PROJECT}.{DATASET}.dim_channels` c
    ON c.channel_id = v.channel_id
  {where}
  GROUP BY 1
)
SELECT d, views, likes, comments
FROM daily
ORDER BY d
"""
daily = run_query(daily_sql, params=params)

st.markdown("### Daily Metrics")
if not daily.empty:
    st.line_chart(daily.set_index("d")[["views", "likes", "comments"]])
else:
    st.info("No time series data available in the selected window.")

# ---------------- Recent top videos ----------------
recent_sql = f"""
SELECT
  v.video_id,
  v.title,
  c.channel_title,
  v.published_at,
  s.view_count    AS views,
  s.like_count    AS likes,
  s.comment_count AS comments
FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
JOIN `{PROJECT}.{DATASET}.dim_videos`   v
  ON v.video_id = s.video_id
JOIN `{PROJECT}.{DATASET}.dim_channels` c
  ON c.channel_id = v.channel_id
{where}
ORDER BY s.view_count DESC
LIMIT 50
"""
recent = run_query(recent_sql, params=params)

st.markdown("### Recent Top Videos")
if not recent.empty:
    cols = ["published_at", "channel_title", "title", "views", "likes", "comments", "video_id"]
    st.dataframe(recent[cols])
else:
    st.info("No videos found for this selection.")

st.markdown("### Engagement Rate (likes + comments per 1000 views)")
ratio_sql = f"""
SELECT
  AVG(SAFE_DIVIDE(like_count + comment_count, view_count) * 1000) AS engagement_per_1k_views
FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
JOIN `{PROJECT}.{DATASET}.dim_videos` v ON v.video_id = s.video_id
JOIN `{PROJECT}.{DATASET}.dim_channels` c ON c.channel_id = v.channel_id
{where}
"""
ratio = run_query(ratio_sql, params=params)
if not ratio.empty:
    st.metric("Avg Engagement / 1 k Views", round(ratio.iloc[0][0], 2))

st.markdown("### Top Channels by Views")
top_channels_sql = f"""
SELECT c.channel_title, SUM(s.view_count) AS total_views
FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
JOIN `{PROJECT}.{DATASET}.dim_videos` v ON v.video_id = s.video_id
JOIN `{PROJECT}.{DATASET}.dim_channels` c ON c.channel_id = v.channel_id
{where}
GROUP BY 1
ORDER BY total_views DESC
LIMIT 10
"""
top_channels = run_query(top_channels_sql, params=params)
if not top_channels.empty:
    st.bar_chart(top_channels.set_index("channel_title")["total_views"])
st.markdown("### Top Channels by Views")
top_channels_sql = f"""
SELECT c.channel_title, SUM(s.view_count) AS total_views
FROM `{PROJECT}.{DATASET}.fact_video_statistics` s
JOIN `{PROJECT}.{DATASET}.dim_videos` v ON v.video_id = s.video_id
JOIN `{PROJECT}.{DATASET}.dim_channels` c ON c.channel_id = v.channel_id
{where}
GROUP BY 1
ORDER BY total_views DESC
LIMIT 10
"""
top_channels = run_query(top_channels_sql, params=params)
if not top_channels.empty:
    st.bar_chart(top_channels.set_index("channel_title")["total_views"])

