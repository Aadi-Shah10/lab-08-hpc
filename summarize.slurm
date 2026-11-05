#!/bin/bash
# Job script for Part 1, Step 2 of README.md: update and complete this script before submitting with sbatch.
# Slurm reads #SBATCH lines only above the first command, so keep them at the top.

# TODO: update the directives below before submitting
#SBATCH --account=<course allocation>
#SBATCH --job-name=<job name>
#SBATCH --partition=<partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=<CPU cores per task>
#SBATCH --mem=<memory per task>
#SBATCH --time=<time limit per task, HH:MM:SS>
#SBATCH --array=<one task per book, numbered from 1 to the number of books>
#SBATCH --output=logs/summarize-%A_%a.out

export HF_HUB_OFFLINE=1 # use the downloaded model in $HF_HOME only, without contacting the Hugging Face Hub

# TODO: load uv

# TODO: run summarize.py with the book list (the first argument) and this task's number
