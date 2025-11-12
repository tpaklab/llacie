#!/bin/bash
echo "*** Inside apptainer_slurm.sh ***" 
date

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

args="$APPTAINER_SCRIPT_VARS"
venv_activate="${args%% *}"    # Get the first space-separated word in args
args="${args#* }"              # Get everything after the first space-separated word
cache_file="${args%% *}"       # Get the second space-separated word in args
args="${args#* }"              # Get everything after the second space-separated word
ids_file="${args%% *}"         # Get the third space-separated word in args
worker_cmd="${args#* }"        # Get everything after the third space-separated word

# Addresses a bug with the outlines package creating a SQLite cache that won't work on NFS mounts
# See: https://github.com/vllm-project/vllm/issues/4193
export OUTLINES_CACHE_DIR="/tmp/.outlines"

if [[ "$cache_file" != "-" ]]; then
    export LLACIE_WORKER_CACHE_PATH="$cache_file"
fi

cd "$REPO_DIR"
source "$venv_activate"
$worker_cmd $(cat "$ids_file")

echo "*** Exiting apptainer_slurm.sh ***"