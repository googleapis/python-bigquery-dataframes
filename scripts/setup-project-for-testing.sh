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
BIGFRAMES_DEFAULT_CONNECTION_NAME=bigframes-default-connection
BIGFRAMES_RF_CONNECTION_NAME=bigframes-rf-conn

if ! test `which gcloud`; then
  echo "gcloud CLI is not installed. Install it from https://cloud.google.com/sdk/docs/install." >&2
  exit 3
fi

function log_and_execute() {
  echo Running command: $*
  $*
}

function ensure_bq_connection_with_iam() {
  if [ $# -ne 2 ]; then
    echo "USAGE: `basename $0` <location> <connection-name>"
    echo "EXAMPLES:"
    echo "       `basename $0` my-project my-connection"
    exit 4
  fi

  location=$1
  connection_name=$2

  log_and_execute bq show \
                    --connection \
                    --project_id=$PROJECT_ID \
                    --location=$location \
                    $connection_name 2>&1 >/dev/null
  if [ $? -ne 0 ]; then
    echo "Connection $connection_name doesn't exists in location \"$location\", creating..."
    log_and_execute bq mk \
                      --connection \
                      --project_id=$PROJECT_ID \
                      --location=$location \
                      --connection_type=CLOUD_RESOURCE \
                      $connection_name
    if [ $? -ne 0 ]; then
      echo "Failed creating connection, exiting."
      exit 5
    fi
  else
    echo "Connection $connection_name already exists in location $location."
  fi

  compact_json_info_cmd="bq show --connection \
                          --project_id=$PROJECT_ID \
                          --location=$location \
                          --format=json \
                          $connection_name"
  compact_json_info_cmd_output=`$compact_json_info_cmd`
  if [ $? -ne 0 ]; then
    echo "Failed to fetch connection info: $compact_json_info_cmd_output"
    exit 6
  fi

  connection_service_account=`echo $compact_json_info_cmd_output | sed -e 's/.*"cloudResource":{"serviceAccountId":"//' -e 's/".*//'`

  # Configure roles for the service accounts associated with the connection
  for role in run.invoker aiplatform.user; do
    log_and_execute gcloud projects add-iam-policy-binding $PROJECT_ID \
                      --member=serviceAccount:$connection_service_account \
                      --role=roles/$role
    if [ $? -ne 0 ]; then
      echo "Failed to set IAM, exiting..."
      exit 7
    fi
  done
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

# Create the default BQ connection in US location
ensure_bq_connection_with_iam "us" "$BIGFRAMES_DEFAULT_CONNECTION_NAME"

# Create commonly used BQ connection in various locations
for location in asia-southeast1 \
                eu \
                europe-west4 \
                southamerica-west1 \
                us \
                us-central1 \
  ; do
  ensure_bq_connection_with_iam "$location" "$BIGFRAMES_RF_CONNECTION_NAME"
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
