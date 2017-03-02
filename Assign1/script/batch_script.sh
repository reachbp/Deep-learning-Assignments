#!/bin/bash
#
#SBATCH --job-name=test_hpc_jitter_dl1
#SBATCH --nodes=4
#SBATCH --tasks-per-node=4
#SBATCH --mem=2GB
#SBATCH --time=04:00:00
 
module purge
module load torch/intel
module load torchvision
module load pytorch/intel 
cd /scratch/pks329/Deep-learning-Assignments/Assign1/script
python  Supervised-augmentations.py > augmentations.out 2>&1
