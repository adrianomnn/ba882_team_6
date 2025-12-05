import streamlit as st
import pandas as pd
from datetime import date
from utils.bq import run_query
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
from groq import Groq
import json
import re

# ---------------- App Page Title -----------------
st.set_page_config(page_title="Weekley Comments Sentiment Analysis", layout="wide")
st.title("Weekley Comments Sentiment Analysis")

# -------------------------------------------
# CONFIG
# -------------------------------------------
PROJECT = st.secrets.get("gcp_project_id") or st.secrets["gcp_service_account"]["project_id"]
DATASET = st.secrets.get("bq_dataset", "youtube_staging")
client = Groq(api_key=st.secrets["gcp_service_account"]["groq_api_key"])

st.markdown(
    """
    Select a week to retrieve all comments collected that week.
    """
)

@st.cache_data(ttl=3600)
def load_available_weeks():
    """
    Returns a DataFrame with columns: year, week, iso_week_label, week_start, week_end
    """
    sql = f"""
    SELECT
      EXTRACT(YEAR FROM DATE(published_at)) AS year,
      EXTRACT(ISOWEEK FROM DATE(published_at)) AS week,
      MIN(DATE(published_at)) AS week_start,
      MAX(DATE(published_at)) AS week_end
    FROM `{PROJECT}.{DATASET}.fact_comments` f
    JOIN `{PROJECT}.{DATASET}.dim_comments` d USING (comment_id)
    WHERE published_at IS NOT NULL
    GROUP BY year, week
    ORDER BY year DESC, week DESC
    """
    df = run_query(sql)
    if df.empty:
        return pd.DataFrame(columns=["year","week","iso_week_label","week_start","week_end"])
    df["week"] = df["week"].astype(int)
    df["year"] = df["year"].astype(int)
    df["iso_week_label"] = df.apply(
        lambda r: f"{r['year']}-W{str(r['week']).zfill(2)} ({r['week_start']} to {r['week_end']})",
        axis=1
    )
    return df

@st.cache_data(ttl=1800)
def load_comments_for_week(year:int, week:int, limit:int = None):
    """
    Returns comments for the given year and iso-week.
    """
    sql = f"""
    SELECT
      f.comment_id,
      d.comment_text,
      f.published_at,
      EXTRACT(YEAR FROM DATE(f.published_at)) as year_extracted,
      EXTRACT(ISOWEEK FROM DATE(f.published_at)) as week_extracted
    FROM `{PROJECT}.{DATASET}.fact_comments` f
    JOIN `{PROJECT}.{DATASET}.dim_comments` d
      ON f.comment_id = d.comment_id
    WHERE EXTRACT(YEAR FROM DATE(f.published_at)) = {year}
    AND EXTRACT(ISOWEEK FROM DATE(f.published_at)) = {week}
    ORDER BY f.published_at
    """
    df = run_query(sql)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"])
    if limit:
        return df.head(limit)
    return df

# --------------------- Vader Sentiment Analysis ---------------------
def apply_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df["vader"] = df["comment"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

def vader_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

# --------------------- LLM Sentiment Analysis ---------------------
def extract_json(text):
    """
    Extracts the first valid JSON object from a messy LLM output.
    Handles markdown fences, text before/after JSON, and multiple JSON objects.
    """
    # 1. Remove markdown code fences
    text = text.strip()
    text = re.sub(r"```(json)?", "", text).strip("` \n")

    # 2. Find JSON object using regex (most reliable)
    json_match = re.search(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, flags=re.DOTALL)
    if not json_match:
        raise ValueError("No JSON object found in LLM output")

    json_str = json_match.group(0)

    # 3. Try parsing
    return json.loads(json_str)

def llm_sentiment_batch(text_batch):
    """
    Analyzes sentiment for a batch of texts.
    Returns exactly len(text_batch) results in the same order.
    """
    prompt = f"""
    You are a sentiment analysis system. For each of the {len(text_batch)} texts below, classify it as positive, neutral, or negative.

    Respond ONLY in valid JSON format with EXACTLY {len(text_batch)} results in the same order:

    {{
    "results": [
        {{"sentiment": "positive|neutral|negative"}},
        {{"sentiment": "positive|neutral|negative"}},
        ...
    ]
    }}

    Do not include the text in your response, only the sentiment labels.
    """

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "\n\n---\n\n".join(text_batch)}
    ]

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=messages,
        temperature=0
    )

    raw_content = response.choices[0].message.content

    try:
        parsed = extract_json(raw_content)
        results = parsed["results"]
        
        # Verify we got exactly the right number of results
        if len(results) != len(text_batch):
            st.warning(f"LLM returned {len(results)} results but expected {len(text_batch)}. Adjusting...")
            # Trim or pad to match
            if len(results) > len(text_batch):
                results = results[:len(text_batch)]
            else:
                # Pad with neutral if we got too few
                while len(results) < len(text_batch):
                    results.append({"sentiment": "neutral"})
        
        return results
    except Exception as e:
        st.error(f"JSON PARSE ERROR: {e}")
        st.code(raw_content, language="text")
        # Return neutral for all as fallback
        return [{"sentiment": "neutral"} for _ in text_batch]

def apply_llm_sentiment(df, batch_size=30):
    """
    Applies LLM sentiment analysis to all comments in the DataFrame.
    Ensures the result matches the DataFrame length exactly.
    """
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0, text="Analyzing sentiments...")

    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        progress_bar.progress(
            batch_num / total_batches, 
            text=f"Processing batch {batch_num}/{total_batches}..."
        )
        
        batch = df["comment_text"].iloc[i:i+batch_size].tolist()
        batch_results = llm_sentiment_batch(batch)
        results.extend(batch_results)
    
    progress_bar.empty()
    
    # Final safety check
    if len(results) != len(df):
        st.error(f"Result mismatch! Got {len(results)} results for {len(df)} comments.")
        st.stop()

    df["llm_sentiment"] = [r["sentiment"] for r in results]
    return df
# --------------------- UI: Week selector ---------------------
weeks_df = load_available_weeks()

if weeks_df.empty:
    st.info("No comments found in BigQuery (fact_comments / dim_comments) yet.")
    st.stop()

# Build options and mapping
options = weeks_df["iso_week_label"].tolist()
default_idx = 0
selected_label = st.selectbox("Select week", options, index=default_idx)

# Map back to numeric year/week
sel_row = weeks_df[weeks_df["iso_week_label"] == selected_label].iloc[0]
selected_year = int(sel_row["year"])
selected_week = int(sel_row["week"])

st.markdown(f"**Selected:** {selected_label}")

# --------------------- Load comments for selection ---------------------
limit = st.number_input("Max comments to preview (0 = all)", min_value=0, step=50, value=200)
limit_arg = None if limit == 0 else int(limit)

with st.spinner("Loading comments for selected week..."):
    comments_df = load_comments_for_week(selected_year, selected_week, limit=limit_arg)

st.markdown("### Comments summary")
if comments_df.empty:
    st.warning("No comments for this week.")
else:
    c1, c2, c3 = st.columns([1, 1, 1.3])
    c1.metric("Comments (rows)", int(len(comments_df)))
    # proportion of comments with non-empty text
    non_empty = comments_df["comment_text"].notna().sum()
    c2.metric("Comments with text", int(non_empty))
    # first / last timestamps
    c3.metric("Date range", f"{comments_df['published_at'].min().date()} → {comments_df['published_at'].max().date()}")

    st.markdown("#### Sample comments")
    st.dataframe(comments_df.reset_index(drop=True).head(200))

    st.markdown("#### Quick text preview (first 10)")
    for idx, row in comments_df.head(10).iterrows():
        st.write(f"- `{row['comment_id']}` — {row['published_at']:%Y-%m-%d %H:%M} — {row['comment_text'][:400]}")


st.subheader("VADER Sentiment Analysis")

if comments_df.empty:
    st.info("No comments found for selected week.")
else:
    # Work on a copy to avoid warnings
    df_week = comments_df.copy()

    # Ensure comment_text is string
    df_week["comment_text"] = df_week["comment_text"].fillna("").astype(str)

    with st.spinner("Running VADER sentiment analysis..."):
        analyzer = SentimentIntensityAnalyzer()
        df_week["vader"] = df_week["comment_text"].apply(
            lambda x: analyzer.polarity_scores(x)["compound"]
        )

        # Apply labels
        def vader_label(score):
            if score >= 0.05:
                return "positive"
            elif score <= -0.05:
                return "negative"
            return "neutral"

        df_week["vader_label"] = df_week["vader"].apply(vader_label)

    st.success("VADER sentiment calculated successfully!")

    # Display sample output
    st.markdown("### Sample Results")
    st.dataframe(df_week[["comment_id", "comment_text", "vader", "vader_label"]].head(20))

    # Summary aggregated metrics
    summary = (
        df_week["vader_label"]
        .value_counts()
        .rename_axis("sentiment")
        .reset_index(name="count")
    )

    st.markdown("### Sentiment Breakdown")
    #st.bar_chart(summary.set_index("sentiment"))

    chart = (
    alt.Chart(summary)
    .mark_bar()
    .encode(
        x=alt.X("sentiment:N", axis=alt.Axis(labelAngle=0)),
        y="count:Q",
        tooltip=["sentiment", "count"]
    )
    .properties(height=350))

    st.altair_chart(chart, use_container_width=True)

    st.subheader("LLM Sentiment Analysis")

if st.button("Run LLM Sentiment"):
    with st.spinner("Analyzing comments with AI..."):
        # Create a copy with VADER results already included
        llm_df = comments_df.copy()
        
        # Make sure we have the vader_label column from earlier
        if "vader_label" not in llm_df.columns:
            analyzer = SentimentIntensityAnalyzer()
            llm_df["vader"] = llm_df["comment_text"].apply(
                lambda x: analyzer.polarity_scores(x)["compound"]
            )
            llm_df["vader_label"] = llm_df["vader"].apply(vader_label)
        
        # Apply LLM sentiment
        llm_df = apply_llm_sentiment(llm_df)
        
        # CRITICAL: Store in session state
        st.session_state["llm_df"] = llm_df

    st.success("LLM sentiment completed!")
    st.dataframe(llm_df[["comment_text", "llm_sentiment", "vader_label"]].head(20))

# Display comparison chart if LLM analysis has been run
if "llm_df" in st.session_state:
    st.markdown("### Comparison: LLM vs VADER")
    
    llm_df = st.session_state["llm_df"]
    
    # Create comparison dataframe
    comp = (
        llm_df[["llm_sentiment", "vader_label"]]
        .rename(columns={"vader_label": "vader"})
        .melt(var_name="model", value_name="sentiment")
        .value_counts()
        .reset_index(name="count")
    )
    
    # Side-by-side bar chart
    chart = (
        alt.Chart(comp)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", axis=alt.Axis(labelAngle=0), title="Sentiment"),
            y=alt.Y("count:Q", title="Count"),
            color=alt.Color("model:N", title="Model"),
            column=alt.Column("model:N", title="Model")
        )
        .properties(width=250, height=300)
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Show confusion matrix style comparison
    st.markdown("### Agreement Analysis")
    
    agreement_df = (
        llm_df.groupby(["vader_label", "llm_sentiment"])
        .size()
        .reset_index(name="count")
    )
    
    # Heatmap showing where models agree/disagree
    heatmap = (
        alt.Chart(agreement_df)
        .mark_rect()
        .encode(
            x=alt.X("vader_label:N", title="VADER Sentiment"),
            y=alt.Y("llm_sentiment:N", title="LLM Sentiment"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
            tooltip=["vader_label", "llm_sentiment", "count"]
        )
        .properties(width=300, height=300)
    )
    
    text = (
        alt.Chart(agreement_df)
        .mark_text(baseline="middle", fontSize=16)
        .encode(
            x="vader_label:N",
            y="llm_sentiment:N",
            text="count:Q"
        )
    )
    
    st.altair_chart(heatmap + text, use_container_width=True)
    
    # Calculate agreement percentage
    total = len(llm_df)
    agreed = (llm_df["vader_label"] == llm_df["llm_sentiment"]).sum()
    agreement_pct = (agreed / total) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Comments", total)
    col2.metric("Models Agree", agreed)
    col3.metric("Agreement Rate", f"{agreement_pct:.1f}%")