import streamlit as st
import pandas as pd
from utils.bq import run_query

st.title("Error Analysis (Latest Snapshot)")

sql = """
SELECT
  p.snapshot_date, p.model_version,
  p.video_id, p.y_prob, p.y_pred, p.y_true,
  v.title, c.channel_title, v.published_at
FROM `ml_artifacts.v_pred_vs_label` p
LEFT JOIN `youtube_staging.dim_videos` v ON v.video_id = p.video_id
LEFT JOIN `youtube_staging.dim_channels` c ON c.channel_id = v.channel_id
"""
df = run_query(sql)

if df.empty:
    st.info("No predictions/labels available yet.")
    st.stop()

df["bucket"] = df.apply(
    lambda r: "TP" if (r.y_pred and r.y_true==True) else
              "FP" if (r.y_pred and r.y_true==False) else
              "FN" if ((not r.y_pred) and r.y_true==True) else "TN",
    axis=1
)
st.subheader("Confusion Buckets")
st.bar_chart(df["bucket"].value_counts())

st.subheader("Biggest False Negatives (high probability but missed)")
fns = df[(~df["y_pred"]) & (df["y_true"]==True)].sort_values("y_prob", ascending=False).head(50)
st.dataframe(fns[["snapshot_date","channel_title","title","y_prob","y_pred","y_true"]])
