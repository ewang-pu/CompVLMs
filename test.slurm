#!/bin/bash 
#SBATCH --job-name=testing         # TODO create a short name for your job
#SBATCH --nodes=1                # node count (default is 1 should work)
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=6G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:40:00          # TODO set total run time limit (HH:MM:SS) (default 1-3 hrs could work)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=evanwang@princeton.edu # TODO add netid if you want emails, or delete these past 3 lines

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.2
conda activate CompIW # TODO your conda environment 

python vision-language-models-are-bows/experiment_scripts/reproduce.py --model_name "openai-clip:ViT-B/32" #"vilt-b32-finetuned-coco" # TODO any bash commands you would run in terminal can go here 