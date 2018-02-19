

gsutil ls -R $GCS_JOB_DIR
MODEL_BINARY=gs://$gcs_job_dir/export/census/<timestamp>/
gcloud ml-engine models create census --regions us-central1
gcloud ml-engine models list
gcloud ml-engine versions list --model census
gcloud ml-engine versions create v1 --model census --origin $MODEL_BINARY --runtime-version 1.4
gcloud ml-engine predict --model census --version v1 --json-instances test.json