#!/bin/bash
#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --partition=lycium
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jp6g18@soton.ac.uk

echo "Starting Job"

module load python/3.6.4
module load cuda/10.2
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${SLURM_SUBMIT_DIR}/src"
python src/main.py tune --trainfile datasets/comp3208-train.csv --testfile datasets/comp3208-test.csv --outputfile predictions.csv

echo "Finishing job"