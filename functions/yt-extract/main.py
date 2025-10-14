import json
import os
import time
import uuid
from datetime import datetime, timezone

import functions_framework
import requests
from dateutil import parser as dtparser
from google.cloud import secretmanager, storage

PROJECT_ID = "qst-ba-882-adam"
VERSION_ID = "latest"

# 建议将你的原始 Bucket 名写在环境变量，部署时 --set-env-vars 传入
RAW_BUCKET = os.environ.get("RAW_BUCKET")  # e.g., "qst-ba-882-adam-youtube-raw"

API_BASE = "https://www.googleapis.com/youtube/v3"

def read_secret(project_id: str, secret_id: str, version: str = "latest") -> str:
    sm = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    resp = sm.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8")

def write_jsonl_to_gcs(bucket_name: str, path: str, rows: list):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(path)
    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    blob.upload_from_string(data, content_type="application/json")
    return f"gs://{bucket_name}/{path}"

def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

def http_get(url, params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@functions_framework.http
def task(request):
    """
    HTTP Params (query/body JSON 都支持)：
      table: one of [videos, video_stats, channels, categories, comments]
      query: 搜索关键词（table=videos 时必填）
      channel_id: 某些场景可用（可选）
      region_code: e.g., US（categories 时常用；可选，默认 US）
      video_ids: 逗号分隔（video_stats、comments 可用）
      max_results: 默认 25
    环境变量：
      RAW_BUCKET: 结果落盘的 GCS bucket 名（不带 gs://）
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.args

    table = (payload.get("table") or "").strip().lower()
    query = (payload.get("query") or "").strip()
    channel_id = (payload.get("channel_id") or "").strip()
    region_code = (payload.get("region_code") or "US").strip()
    video_ids = (payload.get("video_ids") or "").strip()
    max_results = int(payload.get("max_results") or 25)

    if not RAW_BUCKET:
        return (json.dumps({"error": "Missing env RAW_BUCKET"}), 400, {"Content-Type": "application/json"})

    # 读取 YouTube API Key
    try:
        api_key = read_secret(PROJECT_ID, "YOUTUBE_API_KEY", VERSION_ID)
    except Exception as e:
        return (json.dumps({"error": f"Cannot read YOUTUBE_API_KEY: {e}"}), 500, {"Content-Type": "application/json"})

    run_id = now_utc_iso() + "_" + uuid.uuid4().hex[:8]
    rows = []

    try:
        if table == "videos":
            if not query:
                return (json.dumps({"error": "videos requires query"}), 400, {"Content-Type": "application/json"})
            # 1) search.list to get video IDs
            search_params = {
                "key": api_key,
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(max_results, 50),
                "order": "date"
            }
            search = http_get(f"{API_BASE}/search", search_params)
            items = search.get("items", [])
            for it in items:
                snippet = it.get("snippet", {})
                rows.append({
                    "video_id": (it.get("id") or {}).get("videoId"),
                    "channel_id": snippet.get("channelId"),
                    "title": snippet.get("title"),
                    "description": snippet.get("description"),
                    "published_at": snippet.get("publishedAt"),
                    "channel_title": snippet.get("channelTitle"),
                    "search_query": query,
                    "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": f"api:search?q={query}",
                    "run_id": run_id
                })

        elif table == "video_stats":
            # 2) videos.list statistics + contentDetails
            if not video_ids:
                return (json.dumps({"error": "video_stats requires video_ids"}), 400, {"Content-Type": "application/json"})
            params = {
                "key": api_key,
                "part": "snippet,statistics,contentDetails",
                "id": video_ids,
                "maxResults": min(max_results, 50)
            }
            data = http_get(f"{API_BASE}/videos", params)
            for it in data.get("items", []):
                stats = it.get("statistics", {})
                snippet = it.get("snippet", {})
                cd = it.get("contentDetails", {})
                rows.append({
                    "video_id": it.get("id"),
                    "title": snippet.get("title"),
                    "category_id": snippet.get("categoryId"),
                    "duration_iso8601": cd.get("duration"),
                    "views": int(stats.get("viewCount", 0) or 0),
                    "likes": int(stats.get("likeCount", 0) or 0),
                    "comments": int(stats.get("commentCount", 0) or 0),
                    "favorite_count": int(stats.get("favoriteCount", 0) or 0),
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": f"api:videos?id={video_ids}",
                    "run_id": run_id
                })

        elif table == "channels":
            # 3) channels.list
            if not channel_id:
                return (json.dumps({"error": "channels requires channel_id"}), 400, {"Content-Type": "application/json"})
            params = {"key": api_key, "part": "snippet,statistics", "id": channel_id}
            data = http_get(f"{API_BASE}/channels", params)
            for it in data.get("items", []):
                sn = it.get("snippet", {})
                st = it.get("statistics", {})
                rows.append({
                    "channel_id": it.get("id"),
                    "channel_title": sn.get("title"),
                    "country": sn.get("country"),
                    "published_at": sn.get("publishedAt"),
                    "subscriber_count": int(st.get("subscriberCount", 0) or 0),
                    "video_count": int(st.get("videoCount", 0) or 0),
                    "view_count": int(st.get("viewCount", 0) or 0),
                    "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": f"api:channels?id={channel_id}",
                    "run_id": run_id
                })

        elif table == "categories":
            # 4) videoCategories.list
            params = {"key": api_key, "part": "snippet", "regionCode": region_code}
            data = http_get(f"{API_BASE}/videoCategories", params)
            for it in data.get("items", []):
                sn = it.get("snippet", {})
                rows.append({
                    "category_id": it.get("id"),
                    "category_title": sn.get("title"),
                    "assignable": bool(sn.get("assignable", False)),
                    "region": region_code,
                    "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": f"api:videoCategories?regionCode={region_code}",
                    "run_id": run_id
                })

        elif table == "comments":
            # 5) commentThreads.list (top level)
            if not video_ids:
                return (json.dumps({"error": "comments requires video_ids"}), 400, {"Content-Type": "application/json"})
            params = {
                "key": api_key, "part": "snippet", "videoId": video_ids,
                "maxResults": min(max_results, 100), "order": "time", "textFormat": "plainText"
            }
            data = http_get(f"{API_BASE}/commentThreads", params)
            for it in data.get("items", []):
                top = (it.get("snippet") or {}).get("topLevelComment", {})
                tsn = top.get("snippet", {})
                rows.append({
                    "video_id": video_ids,
                    "comment_id": top.get("id"),
                    "author": tsn.get("authorDisplayName"),
                    "text": tsn.get("textDisplay"),
                    "like_count": int(tsn.get("likeCount", 0) or 0),
                    "published_at": tsn.get("publishedAt"),
                    "updated_at": tsn.get("updatedAt"),
                    "ingest_timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_path": f"api:commentThreads?videoId={video_ids}",
                    "run_id": run_id
                })
        else:
            return (json.dumps({"error": "unsupported table"}), 400, {"Content-Type": "application/json"})

        # 落盘到 GCS
        table_name = table
        gcs_path = f"extract/run_id={run_id}/table={table_name}.jsonl"
        uri = write_jsonl_to_gcs(RAW_BUCKET, gcs_path, rows)

        return (json.dumps({
            "message": "extract ok",
            "table": table_name,
            "count": len(rows),
            "gcs_uri": uri,
            "run_id": run_id
        }, ensure_ascii=False), 200, {"Content-Type": "application/json"})

    except Exception as e:
        return (json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"})
