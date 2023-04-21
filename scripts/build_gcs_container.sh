#!/bin/bash

docker build --file ./Dockerfile.gcs -t mklasby/condensed-sparsity:rigl-gcs --shm-size=16gb .
docker push mklasby/condensed-sparsity:rigl-gcs
docker run -itd --name gcp-test --mount 'type=bind,src=/home/mike/.config/gcloud,dst=/root/.config/gcloud' --env-file ./.env.gcs mklasby/condensed-sparsity:rigl-gcs
