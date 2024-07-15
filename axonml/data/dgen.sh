#!/bin/bash

#SBATCH --job-name=dgen
#SBATCH --output=dgen.out
#SBATCH --error=dgen.err
#SBATCH -n 400
#SBATCH -p wmglab
#SBATCH --mem-per-cpu=4G

source ~/.bashrc
module load OpenMPI/2.1.0
conda activate cajal

mpirun --mca io ^ompio \
       --mca mpi_cuda_support 0 \
       -n $SLURM_NTASKS \
       python generate_data.py
