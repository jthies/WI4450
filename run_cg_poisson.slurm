#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
#SBATCH --reservation=subnumatest

# run CG for a n^3 Poisson problem
nx=128
ny=128
nz=128
export OMP_NUM_THREADS=8
srun ./main_benchmarks.x $nx $ny $nz >> results_benchmarks_perf$OMP_NUM_THREADS.out
srun ./main_cg_poisson.x $nx $ny $nz >> results_cg_poisson_perf$OMP_NUM_THREADS.out
srun ./main_cg_poisson_preconditioned.x $nx $ny $nz >> results_cg_poisson_perf_precond$OMP_NUM_THREADS.out

