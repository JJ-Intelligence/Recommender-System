#!/usr/bin/env bash
#PBS -l walltime=16:05:00
#PBS –l nodes=1:ppn=16
#PBS -m ae -M jp6g18@soton.ac.uk
#PBS -N Recommender-System

cd $PBS_O_WORKDIR

module load singularity/3.2.0
export PYTHONPATH="${PYTHONPATH}:/lyceum/jp6g18/git/Recommender-System/src"
singularity exec rs-image.sif python src/main.py tune --trainfile datasets/comp3208-train.csv --testfile datasets/comp3208-test.csv --outputfile predictions.csv