#!/bin/bash
# CUDA version from the command-line argument

cuda=${1:-"3"}
filename=${2:-"main.py"}
config_filename=${3:-"pretrained.yml"}
save_dir=${4:-"saved_files/debug_run/"}

export CUDA_VISIBLE_DEVICES=$cuda

command="python $filename --config_path configs/$config_filename --save_dir $save_dir"
echo "running the following commoand" $command

# Run the command
$command

