#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
#SBATCH --reservation=subnumatest

# run CG for a n^3 Poisson problem
n=1024
export OMP_NUM_THREADS=96
srun ./main_benchmarks.x $n >> results_benchmarks$OMP_NUM_THREADS$n.out
srun ./main_cg_poisson.x $n >> results_cg_poisson$OMP_NUM_THREADS$n.out
