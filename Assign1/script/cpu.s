#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=deepLearning
#SBATCH --mail-type=END
##SBATCH --mail-user=adithyap@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

module load python/intel/2.7.12
# module load python3/intel/3.5.3
module load scikit-learn/intel/0.18.1
module load pytorch/intel/20170125
module load torchvision/0.1.7

# module load python/intel
# module load scikit-learn/intel
# module load pytorch/intel
# module load torchvision
cd /scratch/pks329/Deep-learning-Assignments/Assign1/script
python  Supervised-augmentations.py
# Change path and select filename
