#!/bin/bash

## GET RESOURCES ##

# SBATCH --job-name=build-venv
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G    
#SBATCH --time=1-12:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## RUN SCRIPT ##
./scripts/build_cc_venv.sh
