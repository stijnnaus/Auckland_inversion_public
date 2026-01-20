#!/bin/bash
#SBATCH --job-name="inversion_name_startdate"
#SBATCH --account=nauss
#SBATCH --output="/home/nauss/logs/inversion_name_startdate_rtask.out"
#SBATCH --error="/home/nauss/logs/inversion_name_startdate_rtask.err"
#SBATCH --partition=nesi_prepost
#SBATCH --account=niwa03154
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=192gb
#SBATCH --hint=nomultithread

# Job script to submit on time-chunck of an inversion
python run_inversion.py inversion_name startdate enddate spinup spindown rtask
