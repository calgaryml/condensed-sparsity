#!/bin/bash

## GET RESOURCES ##

#SBATCH --job-name=get-imagenet-data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-12:00:00
#SBATCH --mail-user=mklasby@ucalgary.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=def-yani

## RUN SCRIPT ##

mkdir ILSVRC && cd ./ILSVRC
wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget  https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
