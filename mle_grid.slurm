#!/bin/bash
#SBATCH --job-name=mle_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=10G

#SBATCH --array=1-144

singularity exec --overlay /scratch/cd2754/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate
# python ./mle.py --task_id $SLURM_ARRAY_TASK_ID --name mle_oneneuron_norm2 --target_neurons 1 --neurons 500 --mcmc_samples 300 --seed 12 --alpha 0.1 --gd_updates_per_iter 1 --niter 601 --ball_radius 2.

# python ./mle.py --task_id $SLURM_ARRAY_TASK_ID --name mle_oneneuron_norm10 --target_neurons 1 --neurons 500 --mcmc_samples 300 --seed 12 --alpha 0.1 --gd_updates_per_iter 1 --niter 601 --ball_radius 10.

python ./mle.py --task_id $SLURM_ARRAY_TASK_ID --name twoneuron_norm5 --target_neurons 2 --neurons 500 --mcmc_samples 300 --allow_negative_weights --seed 12 --alpha 0.1 --gd_updates_per_iter 1 --niter 1001 --n_negative_weights 2 --ball_radius 5.

python ./mle.py --task_id $SLURM_ARRAY_TASK_ID --name twoneuron_norm10 --target_neurons 2 --neurons 500 --mcmc_samples 300 --allow_negative_weights --seed 12 --alpha 0.1 --gd_updates_per_iter 1 --niter 1001 --n_negative_weights 2 --ball_radius 10.

python ./mle.py --task_id $SLURM_ARRAY_TASK_ID --name fourneuron_norm30 --target_neurons 4 --neurons 500 --mcmc_samples 300 --allow_negative_weights --seed 12 --alpha 0.1 --ball_radius 30. --gd_updates_per_iter 1 --niter 1001 --n_negative_weights 2


"
