#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450

# run CG for a n^3 Poisson problem
n=600
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
srun ./main_cg_poisson.x $n