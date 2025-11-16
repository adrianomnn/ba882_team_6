#!/bin/bash

# ======================================================
# Configure YOUR project
# ======================================================

gcloud config set project adrineto-qst882-fall25

echo "Checking if Secret Manager key exists..."

if ! gcloud secrets describe YOUTUBE_API_KEY --project=adrineto-qst882-fall25 > /dev/null 2>&1; then
    echo "Secret 'YOUTUBE_API_KEY' not found in Secret Manager!"
    exit 1
else
    echo "Secret 'YOUTUBE_API_KEY' found in Secret Manager."
fi

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
echo "Deploying the YouTube data transformer"
echo "======================================================"

gcloud functions deploy raw-transform \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point task \
    --source ./raw-transform \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512MB 

echo "======================================================"
echo "Deploying the YouTube MLOPs "
echo "======================================================"

gcloud functions deploy mlops-pipeline \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point http_entry \
    --source ./mlops-pipeline \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512MB

echo "======================================================"
echo "Deploying the YouTube MLOPs Parallel"
echo "======================================================"

gcloud functions deploy mlops-pipeline-parallel \
    --gen2 \
    --runtime ${RUNTIME} \
    --trigger-http \
    --entry-point train_model \
    --source ./mlops-pipeline-parallel \
    --stage-bucket ${STAGE_BUCKET} \
    --service-account ${SERVICE_ACCOUNT} \
    --region ${REGION} \
    --allow-unauthenticated \
    --timeout 540s \
    --memory 2048MB \

echo "======================================================"
echo "YouTube pipeline deployment completed successfully!"
echo "======================================================"
