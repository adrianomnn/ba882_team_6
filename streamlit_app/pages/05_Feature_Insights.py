import streamlit as st
from utils.bq import run_query

st.title("Feature Insights (Gap-aware + Engagement)")

PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]

with st.sidebar:
    ndays = st.slider("Lookback window (days)", 7, 90, 30)

# engagement features
eng_sql = f"""
SELECT snapshot_date,
       AVG(engagement_rate) AS avg_engagement
FROM `{PROJECT}.ms_golden.video_engagement_features`
WHERE snapshot_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @d DAY)
GROUP BY snapshot_date
ORDER BY snapshot_date
"""
eng = run_query(eng_sql, params={"d":str(ndays)})

# comment activity features (volume proxy)
cm_sql = f"""
SELECT snapshot_date,
       AVG(comment_volume) AS avg_comment_volume
FROM `{PROJECT}.ms_golden.video_comment_activity_features`
WHERE snapshot_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @d DAY)
GROUP BY snapshot_date
ORDER BY snapshot_date
"""
cm = run_query(cm_sql, params={"d":str(ndays)})

# gap-aware deltas from your ml features table
ml_sql = f"""
SELECT snapshot_date,
       AVG(views_delta_7d)     AS avg_views_delta_7d,
       AVG(likes_delta_7d)     AS avg_likes_delta_7d,
       AVG(comments_delta_7d)  AS avg_comments_delta_7d
FROM `{PROJECT}.ms_golden.video_features_for_ml`
WHERE snapshot_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @d DAY)
GROUP BY snapshot_date
ORDER BY snapshot_date
"""
mlf = run_query(ml_sql, params={"d":str(ndays)})

st.subheader("Engagement Rate (avg, daily)")
if not eng.empty: st.line_chart(eng.set_index("snapshot_date")["avg_engagement"])
else: st.info("No engagement data in range.")

st.subheader("Comment Volume (avg, daily)")
if not cm.empty: st.line_chart(cm.set_index("snapshot_date")["avg_comment_volume"])

st.subheader("Gap-aware Deltas (avg, daily)")
if not mlf.empty:
    st.line_chart(mlf.set_index("snapshot_date")[["avg_views_delta_7d","avg_likes_delta_7d","avg_comments_delta_7d"]])
