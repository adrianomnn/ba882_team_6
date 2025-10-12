#!/bin/bash

# ======================================================
# Configure YOUR project
# ======================================================

gcloud config set project adrineto-qst882-fall25

if [ -z "$YOUTUBE_API_KEY" ]; then
    echo "ERROR: YOUTUBE_API_KEY environment variable is not set!"
    echo "Please set it with: export YOUTUBE_API_KEY='your-api-key'"
    exit 1
fi

echo "YOUTUBE_API_KEY is set (length: ${#YOUTUBE_API_KEY})"

# Variables
REGION="us-central1"
RUNTIME="python312"
SERVICE_ACCOUNT="class3demosa@adrineto-qst882-fall25.iam.gserviceaccount.com"
STAGE_BUCKET="adrineto-ba882-fall25-functions"

# ======================================================
# Deploy schema setup function
# ======================================================

echo "======================================================"
echo "Deploying the YouTube raw schema setup"
echo "======================================================"

gcloud functions deploy raw-schema \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point task \
    --source ./raw-schema \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512MB 

# ======================================================
# Deploy YouTube data extractor
# ======================================================

echo "======================================================"
echo "Deploying the YouTube data extractor"
echo "======================================================"

gcloud functions deploy raw-extract \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point task \
    --source ./raw-extract \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512MB \
    --set-env-vars YOUTUBE_API_KEY=$YOUTUBE_API_KEY

# ======================================================
# Deploy YouTube data loader (GCS → MotherDuck)
# ======================================================

echo "======================================================"
echo "Deploying the YouTube data loader"
echo "======================================================"

gcloud functions deploy raw-parse \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point task \
    --source ./raw-parse \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "YouTube pipeline deployment completed successfully!"
echo "======================================================"
