#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
#SBATCH --reservation=subnumatest

# run CG for a n^3 Poisson problem
n=128
export OMP_NUM_THREADS=24
srun ./main_benchmarks.x $n >> benchmarks$OP_NUM_THREADS.out
srun ./main_cg_poisson.x $n >> cg_poisson$OP_NUM_THREADS.out
