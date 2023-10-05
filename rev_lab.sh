#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=0
#SBATCH --time=0-12:00
#SBATCH --account=def-wanglab-ab
#SBATCH --output=/scratch/alif/oft/outputs/oft-lab-%j.out

module load python/3.10
module load gcc/9.3.0 arrow
source $SCRATCH/reversal/bin/activate
srun $VIRTUAL_ENV/bin/jupyterlab.sh

