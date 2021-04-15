# ebms_shallow_nn
Code for the paper "On Energy-Based Models with Overparametrized Shallow Neural Networks"

EBMKLSGD.m contains the Matlab code for maximum likelihood training in 3 dimensions. KLfeatzunnorm1.mp4 contains the movie of the training dynamics in 3 dimensions.

To run maximum likelihood, KSD and F1SD experiments in arbitrary dimensions in an HPC cluster, the steps are:

1) Run sbatch mle_grid.slurm, sbatch ksd_grid.slurm, sbatch f1_sd_grid.slurm (resp.) with the desired setting. This runs a parameter grid and stores the results for each setting as a subfolders in a folder named res. 

2) Run figures.py with CLEAR_BEST and SAVE_BEST set to True in lines 13 and 14, and show_reps set to False in line 185. This will output figures without error bars in the figures folder and store the best hyperparameter configuration for each setting in files named hyper.txt, in each of the subfolders of res.

3) Run sbatch rep_mle_grid.slurm, sbatch rep_ksd_grid.slurm, sbatch rep_f1_sd_grid.slurm (resp.) with the desired setting. For the setting chosen, this runs ten jobs for the optimal hyperparameter configuration found in the hyper.txt file. 

2) Run figures.py with CLEAR_BEST and SAVE_BEST set to False in lines 13 and 14, and show_reps set to False in line 185. This will output figures with error bars in the figures folder.