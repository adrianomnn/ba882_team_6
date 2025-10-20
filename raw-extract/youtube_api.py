"""
YouTube API wrapper functions
"""

import os
import pandas as pd
from googleapiclient.discovery import build
from datetime import datetime
from googleapiclient.errors import HttpError
from google.cloud import secretmanager

# settings
project_id = 'adrineto-qst882-fall25'
secret_id = 'YOUTUBE_API_KEY'
version_id = 'latest'

# Global variable to cache the YouTube client
_youtube_client = None


def get_youtube_client():
    """
    Lazy initialization of YouTube API client.
    Only creates client when first called, not at import time.
    """
    global _youtube_client
    
    if _youtube_client is None:
        sm = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = sm.access_secret_version(request={"name": name})
        API_KEY = response.payload.data.decode("UTF-8")
        
        if not API_KEY:
            raise ValueError("YOUTUBE_API_KEY environment variable not set!")
        
        _youtube_client = build('youtube', 'v3', developerKey=API_KEY)
        print("YouTube API client initialized")
    
    return _youtube_client


def get_video(query, max_results=50, order='relevance'):
    """
    Search for videos by keyword.
    Returns DataFrame with video metadata.
    """
    try:
        youtube = get_youtube_client()
        response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=min(max_results, 50),
            type='video',
            order='date'
        ).execute()

        videos = []
        for item in response.get('items', []):
            snippet = item['snippet']
            videos.append({
                'video_id': item['id']['videoId'],
                'channel_id': snippet['channelId'],
                'title': snippet['title'],
                'description': snippet['description'],
                'published_at': snippet['publishedAt'],
                'search_query': query,
                'search_order': order
            })
        
        return pd.DataFrame(videos)
    
    except Exception as e:
        print(f"Error in get_video: {str(e)}")
        return pd.DataFrame()


def get_channel_details(channel_ids):
    """
    Retrieve basic channel information for a list of channel IDs.
    Returns DataFrame with channel metadata.
    """
    if not channel_ids:
        return pd.DataFrame()
    
    try:
        youtube = get_youtube_client()
        # API accepts max 50 IDs per request
        all_channels = []
        for i in range(0, len(channel_ids), 50):
            batch = channel_ids[i:i+50]
            response = youtube.channels().list(
                part="snippet,statistics",
                id=",".join(batch)
            ).execute()

            for item in response.get('items', []):
                snippet = item['snippet']
                stats = item['statistics']
                all_channels.append({
                    'channel_id': item['id'],
                    'channel_title': snippet['title'],
                    'channel_description': snippet.get('description'),
                    'country': snippet.get('country'),
                    'published_at': snippet['publishedAt'],
                    'subscriber_count': int(stats.get('subscriberCount', 0)),
                    'video_count': int(stats.get('videoCount', 0)),
                    'view_count': int(stats.get('viewCount', 0))
                })
        
        return pd.DataFrame(all_channels)
    
    except Exception as e:
        print(f"Error in get_channel_details: {str(e)}")
        return pd.DataFrame()


def get_video_statistics(video_ids):
    """
    Retrieve engagement metrics for videos.
    Returns DataFrame with video statistics.
    """
    if not video_ids:
        return pd.DataFrame()
    
    try:
        youtube = get_youtube_client()
        all_stats = []
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            response = youtube.videos().list(
                part="statistics,snippet,contentDetails",
                id=",".join(batch)
            ).execute()

            for item in response.get('items', []):
                stats = item['statistics']
                snippet = item['snippet']
                details = item['contentDetails']

                all_stats.append({
                    'video_id': item['id'],
                    'category_id': snippet.get('categoryId'),
                    'duration': details.get('duration'),
                    'view_count': int(stats.get('viewCount', 0)),
                    'like_count': int(stats.get('likeCount', 0)),
                    'comment_count': int(stats.get('commentCount', 0)),
                    'tags': ','.join(snippet.get('tags', [])) if snippet.get('tags') else None,
                    'favorite_count': int(stats.get('favoriteCount', 0)),
                    'collected_at': datetime.utcnow().isoformat()
                })
        
        return pd.DataFrame(all_stats)
    
    except Exception as e:
        print(f"Error in get_video_statistics: {str(e)}")
        return pd.DataFrame()

def get_video_comments(video_id, max_comments=50):
    """
    Retrieve top-level comments for a video.
    Returns DataFrame with comment data, or None if comments are disabled.
    """
    if not video_id:
        return None
    
    try:
        youtube = get_youtube_client()
        comments = []
        next_page_token = None
        total_fetched = 0

        while total_fetched < max_comments:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - total_fetched),
                pageToken=next_page_token,
                textFormat="plainText"
            ).execute()

            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_id": item["id"],
                    "author_display_name": snippet.get("authorDisplayName"),
                    "text_display": snippet.get("textDisplay"),
                    "like_count": snippet.get("likeCount", 0),
                    "published_at": snippet.get("publishedAt"),
                })

            total_fetched += len(response.get("items", []))
            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break
        
        return pd.DataFrame(comments) if comments else None
    
    except HttpError as e:
        error_json = e.content.decode("utf-8")
        if "commentsDisabled" in error_json:
            print(f"Comments disabled for video {video_id}")
            return None
        else:
            print(f"HttpError fetching comments for {video_id}: {error_json}")
            return None

    except Exception as e:
        print(f"Unexpected error fetching comments for {video_id}: {e}")
        return None


def get_video_categories(region_code="US"):
    """
    Retrieve video categories for a specific region.
    Returns DataFrame mapping category_id to category_name.
    """
    try:
        youtube = get_youtube_client()
        response = youtube.videoCategories().list(
            part="snippet",
            regionCode=region_code
        ).execute()

        categories = []
        for item in response.get("items", []):
            snippet = item["snippet"]
            categories.append({
                "category_id": item["id"],
                "category_title": snippet["title"],
                "assignable": snippet["assignable"],
                "region": region_code
            })
        
        return pd.DataFrame(categories)
    
    except Exception as e:
        print(f"Error in get_video_categories: {str(e)}")
        return pd.DataFrame()