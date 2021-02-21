#!/usr/bin/env bash
#PBS -l walltime=12:00:00
#PBS –l nodes=1:ppn=16
#PBS -m ae -M jp6g18@soton.ac.uk

module load singularity/3.2.0
export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/Recommender-System/src"
singularity exec image.sif python src/main.py --trainfile datasets/comp3208-train.csv --testfile datasets/comp3208-test.csv --outputfile predictions.csv