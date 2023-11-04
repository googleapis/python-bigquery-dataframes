#!/bin/bash

if [ $# -ne 1 ]; then
  echo "USAGE: `basename $0` <project-name>"
  exit 2
fi

PROJECT=$1


if ! test `which gcloud`; then
  echo "gcloud CLI is not installed. Install it from https://cloud.google.com/sdk/docs/install." >&2
  exit 3
fi

for service in aiplatform.googleapis.com \
               bigquery.googleapis.com \
               bigqueryconnection.googleapis.com \
               bigquerystorage.googleapis.com \
               cloudfunctions.googleapis.com \
               cloudresourcemanager.googleapis.com \
               run.googleapis.com \
    ; do
    gcloud --project=$PROJECT --verbosity=debug services enable $service
done
