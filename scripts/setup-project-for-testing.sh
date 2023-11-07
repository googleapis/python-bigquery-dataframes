#!/bin/bash

if [ $# -ne 1 ]; then
  echo "USAGE: `basename $0` <project-id> [<principal>]"
  echo "EXAMPLES:"
  echo "       `basename $0` my-project"
  echo "       `basename $0` my-project user:user_id@example.com"
  echo "       `basename $0` my-project group:group_id@example.com"
  echo "       `basename $0` my-project serviceAccount:service_account_id@example.com"
  exit 2
fi

PROJECT_ID=$1
PRINCIPAL=$2
BIGFRAMES_RF_CONNECTION_NAME=bigframes-rf-conn

if ! test `which gcloud`; then
  echo "gcloud CLI is not installed. Install it from https://cloud.google.com/sdk/docs/install." >&2
  exit 3
fi

function log_and_execute() {
  echo Running command: $*
  $*
}

# Enable APIs
for service in aiplatform.googleapis.com \
               bigquery.googleapis.com \
               bigqueryconnection.googleapis.com \
               bigquerystorage.googleapis.com \
               cloudbuild.googleapis.com \
               cloudfunctions.googleapis.com \
               cloudresourcemanager.googleapis.com \
               run.googleapis.com \
  ; do
  log_and_execute gcloud --project=$PROJECT_ID services enable $service
done

# Create BQ connections
for location in asia-southeast1 \
                eu \
                europe-west4 \
                southamerica-west1 \
                us \
                us-central1 \
  ; do
  log_and_execute bq show \
                    --connection \
                    --project_id=$PROJECT_ID \
                    --location=$location \
                    $BIGFRAMES_RF_CONNECTION_NAME 2>&1 >/dev/null
  if [ $? -ne 0 ]; then
    log_and_execute bq mk \
                      --connection \
                      --project_id=$PROJECT_ID \
                      --location=$location \
                      --connection_type=CLOUD_RESOURCE \
                      $BIGFRAMES_RF_CONNECTION_NAME
  else
    echo "Connection $BIGFRAMES_RF_CONNECTION_NAME already exists in location $location."
  fi

  compact_json_info_cmd="bq show --connection \
                          --project_id=$PROJECT_ID \
                          --location=$location \
                          --format=json \
                          $BIGFRAMES_RF_CONNECTION_NAME"
  connection_service_account=`$compact_json_info_cmd | sed -e 's/.*"cloudResource":{"serviceAccountId":"//' -e 's/".*//'`

  # Configure roles for the service accounts associated with the connection
  for role in run.invoker aiplatform.user; do
    log_and_execute gcloud projects add-iam-policy-binding $PROJECT_ID \
                      --member=serviceAccount:$connection_service_account \
                      --role=roles/$role
  done
done

# Set up IAM roles for principal
if [ "$PRINCIPAL" != "" ]; then
  for role in aiplatform.user \
              bigquery.user \
              bigquery.connectionAdmin \
              bigquery.dataEditor \
              browser \
              cloudfunctions.developer \
              iam.serviceAccountUser \
    ; do
    log_and_execute gcloud projects add-iam-policy-binding $PROJECT_ID --member=$PRINCIPAL --role=roles/$role
  done
fi
