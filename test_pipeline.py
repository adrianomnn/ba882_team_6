#!/usr/bin/env python3
"""
Test script to trigger the YouTube data pipeline
Run this after deploying your Cloud Functions
"""

import requests
import json
import time

# Configuration
PROJECT_ID = "adrineto-qst882-fall25"
REGION = "us-central1"
BASE_URL = f"https://{REGION}-{PROJECT_ID}.cloudfunctions.net"

# Function URLs
SCHEMA_URL = f"{BASE_URL}/raw-schema"
EXTRACT_URL = f"{BASE_URL}/raw-extract"
PARSE_URL = f"{BASE_URL}/raw-parse"


def test_schema_creation():
    """Step 1: Create schema in BigQuery"""
    print("=" * 60)
    print("Step 1: Creating Schema in BigQuery")
    print("=" * 60)
    
    try:
        response = requests.get(SCHEMA_URL, timeout=120)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✅ Schema created successfully!\n")
            return True
        else:
            print("❌ Schema creation failed!\n")
            return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def test_data_extraction(query="data engineering"):
    """Step 2: Extract YouTube data"""
    print("=" * 60)
    print("Step 2: Extracting YouTube Data")
    print("=" * 60)
    print(f"Query: {query}")
    
    try:
        response = requests.get(
            EXTRACT_URL,
            params={"query": query},
            timeout=300  # 5 minutes timeout for API calls
        )
        print(f"Status Code: {response.status_code}")
        
        # Print raw response for debugging
        print(f"Raw Response Text: {response.text[:500]}")  # First 500 chars
        
        try:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        except json.JSONDecodeError as je:
            print(f"❌ Failed to parse JSON response: {je}")
            print(f"Raw response: {response.text}")
            return None
        
        if response.status_code == 200:
            print("✅ Data extracted successfully!\n")
            return result
        else:
            print("❌ Extraction failed!\n")
            print(f"Error details: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading(extraction_result):
    """Step 3: Load data to BigQuery"""
    print("=" * 60)
    print("Step 3: Loading Data to BigQuery")
    print("=" * 60)
    
    payload = {
        "bucket_name": extraction_result["bucket_name"],
        "blob_name": extraction_result["blob_name"],
        "run_id": extraction_result["run_id"]
    }
    
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(
            PARSE_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        print(f"Status Code: {response.status_code}")
        
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200 and result.get("status") == "success":
            print("✅ Data loaded successfully!\n")
            return True
        else:
            print("❌ Loading failed!\n")
            return False
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False


def main():
    """Run the complete pipeline"""
    print("\n" + "=" * 60)
    print("🚀 YouTube Data Pipeline Test")
    print("=" * 60 + "\n")
    
    # Step 1: Schema
    schema_success = test_schema_creation()
    if not schema_success:
        print("⚠️ Schema creation failed, but continuing anyway...")
    
    time.sleep(2)
    
    # Step 2: Extract
    extraction_result = test_data_extraction(query="data engineering")
    if not extraction_result:
        print("❌ Pipeline failed at extraction step")
        return
    
    time.sleep(2)
    
    # Step 3: Load
    load_success = test_data_loading(extraction_result)
    if not load_success:
        print("❌ Pipeline failed at loading step")
        return
    
    # Summary
    print("=" * 60)
    print("🎉 Pipeline Completed Successfully!")
    print("=" * 60)
    print(f"\n✅ Data Location: gs://{extraction_result['bucket_name']}/{extraction_result['blob_name']}")
    print(f"✅ Run ID: {extraction_result['run_id']}")
    print(f"✅ Videos Processed: {extraction_result.get('videos_count', 'N/A')}")
    print(f"✅ Channels Processed: {extraction_result.get('channels_count', 'N/A')}")
    print(f"✅ Comments Processed: {extraction_result.get('comments_count', 'N/A')}")
    print("\nNext Steps:")
    print("1. Check GCS bucket for raw data")
    print("2. Check MotherDuck for loaded data")
    print("3. View logs in GCP Console")


if __name__ == "__main__":
    main()