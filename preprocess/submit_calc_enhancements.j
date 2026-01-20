#!/bin/bash -l
#SBATCH --job-name=calc_enh
#SBATCH --account=nauss
#SBATCH --output=/home/nauss/logs/submit_calc_enhancements.out
#SBATCH --error=/home/nauss/logs/submit_calc_enhancements.err
#SBATCH --partition=nesi_prepost
#SBATCH --account=niwa03154
#SBATCH --time=18:00:00
#SBATCH --ntasks=14
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=14gb

conda activate /nesi/project/niwa03154/CARBONWATCH/nauss_conda/miniconda3/envs/daemon
cd /home/nauss/Code/Auckland_inversion/preprocess/


python calc_enhancements_conv_to_inv_grid.py 2021 12 &
python calc_enhancements_conv_to_inv_grid.py 2023 1 &

year=2022
for i in {1..12}
do
  echo "Preprocessing ${year} / ${i}"
  python calc_enhancements_conv_to_inv_grid.py ${year} ${i} &
done

wait
