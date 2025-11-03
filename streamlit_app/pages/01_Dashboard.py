import streamlit as st
import pandas as pd
from utils import bq

st.set_page_config(page_title="📊 Dashboard", layout="wide")
st.title("📊 Dashboard")

DATASET = st.secrets.get("bq_dataset", "youtube_staging") if hasattr(st, "secrets") else "youtube_staging"

# Channel filter options
channels_df = bq.run_query(f"""
SELECT DISTINCT channel_title
FROM `{DATASET}.dim_videos`
ORDER BY 1
""")
channel_options = ["(All)"]
if channels_df is not None and not channels_df.empty:
    channel_options += [c for c in channels_df["channel_title"].dropna().astype(str).tolist() if c]
else:
    st.info("Table youtube_staging.dim_videos not found or empty; channel filter limited to (All).")

with st.sidebar:
    ndays_choice = st.radio("Date range", options=["Last 7 days", "Last 30 days", "Last 90 days"], index=1)
    ndays = 30
    if ndays_choice == "Last 7 days":
        ndays = 7
    elif ndays_choice == "Last 30 days":
        ndays = 30
    elif ndays_choice == "Last 90 days":
        ndays = 90

    channel = st.selectbox("Channel", options=channel_options, index=0)

params = {"ndays": ndays}
channel_clause = ""
if channel and channel != "(All)":
    channel_clause = " AND dv.channel_title = @channel"
    params["channel"] = channel

# KPIs: Total Views, Likes, Comments
kpi_sql = f"""
WITH base AS (
  SELECT
    fs.publish_time,
    fs.views,
    fs.likes,
    fs.comments
  FROM `{DATASET}.fact_video_statistics` fs
  JOIN `{DATASET}.dim_videos` dv USING (video_id)
  WHERE DATE(fs.publish_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL @ndays DAY)
  {channel_clause}
)
SELECT
  SUM(views) AS total_views,
  SUM(likes) AS total_likes,
  SUM(comments) AS total_comments
FROM base
"""

kpi_df = bq.run_query(kpi_sql, params=params)
col1, col2, col3 = st.columns(3)
if kpi_df is not None and not kpi_df.empty:
    with col1:
        st.metric("Total Views", f"{int(kpi_df['total_views'].iloc[0] or 0):,}")
    with col2:
        st.metric("Total Likes", f"{int(kpi_df['total_likes'].iloc[0] or 0):,}")
    with col3:
        st.metric("Total Comments", f"{int(kpi_df['total_comments'].iloc[0] or 0):,}")
else:
    st.info("Create youtube_staging.fact_video_statistics and youtube_staging.dim_videos with data to populate KPIs.")

# Daily line chart
series_sql = f"""
SELECT
  DATE(fs.publish_time) AS dt,
  SUM(fs.views) AS views,
  SUM(fs.likes) AS likes,
  SUM(fs.comments) AS comments
FROM `{DATASET}.fact_video_statistics` fs
JOIN `{DATASET}.dim_videos` dv USING (video_id)
WHERE DATE(fs.publish_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL @ndays DAY)
{channel_clause}
GROUP BY dt
ORDER BY dt
"""
series_df = bq.run_query(series_sql, params=params)
if series_df is not None and not series_df.empty:
    series_df = series_df.sort_values("dt")
    st.line_chart(series_df.set_index("dt")[["views", "likes", "comments"]])
else:
    st.info("No time series data available in the selected window.")

# Recent Top 20 videos
table_sql = f"""
SELECT
  fs.video_id,
  dv.title,
  dv.channel_title,
  fs.publish_time,
  fs.views
FROM `{DATASET}.fact_video_statistics` fs
JOIN `{DATASET}.dim_videos` dv USING (video_id)
WHERE DATE(fs.publish_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL @ndays DAY)
{channel_clause}
ORDER BY fs.views DESC
LIMIT 20
"""
recent_df = bq.run_query(table_sql, params=params)
st.subheader("Recent Top Videos")
if recent_df is not None and not recent_df.empty:
    st.dataframe(recent_df)
else:
    st.info("No recent videos found in the selected window.")

# Optional Top Players section
# Check existence via INFORMATION_SCHEMA
exists_sql = f"""
SELECT table_name
FROM `{DATASET}.INFORMATION_SCHEMA.TABLES`
WHERE table_name IN ('fact_player_mentions','dim_players')
"""
exists_df = bq.run_query(exists_sql)
has_mentions = False
if exists_df is not None and not exists_df.empty:
    names = set(exists_df["table_name"].tolist())
    has_mentions = {'fact_player_mentions', 'dim_players'}.issubset(names)

st.subheader("Top Players")
if has_mentions:
    players_sql = f"""
    WITH player_views AS (
      SELECT
        pm.player_id,
        SUM(fs.views) AS total_views
      FROM `{DATASET}.fact_player_mentions` pm
      JOIN `{DATASET}.fact_video_statistics` fs USING (video_id)
      JOIN `{DATASET}.dim_videos` dv USING (video_id)
      WHERE DATE(fs.publish_time) >= DATE_SUB(CURRENT_DATE(), INTERVAL @ndays DAY)
      {channel_clause}
      GROUP BY pm.player_id
    )
    SELECT p.full_name, pv.total_views
    FROM player_views pv
    JOIN `{DATASET}.dim_players` p ON p.player_id = pv.player_id
    ORDER BY pv.total_views DESC
    LIMIT 20
    """
    players_df = bq.run_query(players_sql, params=params)
    if players_df is not None and not players_df.empty:
        st.dataframe(players_df)
    else:
        st.info("No player mentions found in the selected window.")
else:
    st.info("Player mentions tables not found; skipping this section.")
