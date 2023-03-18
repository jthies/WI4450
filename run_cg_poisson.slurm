#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
#SBATCH --reservation=subnumatest

# run CG for a n^3 Poisson problem
nx=600
ny=600
nz=600
export OMP_NUM_THREADS=12
srun ./main_benchmarks.x $nx $ny $nz >> results_benchmarks_flops$nx$OMP_NUM_THREADS.out
