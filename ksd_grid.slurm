#!/bin/bash
#SBATCH --job-name=ksd_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=24G

#SBATCH --array=1-144

singularity exec --overlay /scratch/cd2754/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate
# python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksdmetro --seed 43
# python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksdmetro_norm10 --ball_radius 10. --seed 43

# python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksd_twoneuron_norm5 --target_neurons 2 --neurons 500 --allow_negative_weights --seed 12 --alpha 0.1 --niter 1001 --n_negative_weights 2 --ball_radius 5.

# python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksd_twoneuron_norm10 --target_neurons 2 --neurons 500 --allow_negative_weights --seed 12 --alpha 0.1 --niter 1001 --n_negative_weights 2 --ball_radius 10.

# python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksd_fourneuron_norm30 --target_neurons 4 --neurons 500 --allow_negative_weights --seed 12 --alpha 0.1 --niter 1001 --n_negative_weights 2 --ball_radius 30.

python ./ksd.py --task_id $SLURM_ARRAY_TASK_ID --name ksd_oneneuron_norm10 --target_neurons 1 --neurons 500 --seed 12 --alpha 0.1 --niter 1001 --ball_radius 10.
"
