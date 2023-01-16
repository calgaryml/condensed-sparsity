#!/bin/bash

gcloud auth login
gcloud auth application-default login
gcloud config set project $GCP_PROJECT
