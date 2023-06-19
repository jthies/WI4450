#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=education-eemcs-courses-summerschool2023
#SBATCH --reservation=summerschool2023
#SBATCH --time=00:10:00

export OMP_NUM_THREADS=16
srun ./main.x >> output_simulations-${OMP_NUM_THREADS}.out

