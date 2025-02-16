#!/bin/bash

# Set number of threads
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# General configurations
export CVD=4
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export VS="self2"
export SPG=6
export BS=40
export NB=5
export TIME_LIMIT=3600
export CONFIG_NAME="wyck_prim_new_self"

# Change to working directory
cd /home/holywater2/crystal_gen/mattergen

# Function to generate and evaluate structures
generate_and_evaluate() {
    local gs=$1
    local ge=$2
    local ts=$3
    local te=$4

    export RESULTS_PATH="results/${VS}/spg${SPG}_gs${gs}_ge${ge}_ts${ts}_te${te}_${VS}"

    echo "Running generation for gs=${gs}, ge=${ge}, ts=${ts}, te=${te}..."

    # Generate batch_size * num_batches samples
    mkdir -p $RESULTS_PATH
    CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py \
        $RESULTS_PATH \
        $MODEL_PATH \
        --project_save True \
        --batch_size=$BS --num_batches $NB \
        --sampling_config_name $CONFIG_NAME \
        --sampling_config_overrides="[\"condition_loader_partial.space_groups=${SPG}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\",\"sampler_partial.ts=$ts\",\"sampler_partial.te=$te\"]" \
    | tee $RESULTS_PATH/gen.log
    mv $RESULTS_PATH/gen.log $RESULTS_PATH/result_0/gen.log

    echo "Running evaluation..."

    # Evaluate generated structures
    CUDA_VISIBLE_DEVICES=$CVD timeout $TIME_LIMIT uv run python scripts/evaluate.py \
        --structures_path="$RESULTS_PATH/result_0/generated_crystals_cif" \
        --relax=True --structure_matcher='disordered' \
        --save_as="$RESULTS_PATH/result_0/metrics.json" > $RESULTS_PATH/result_0/eval.log &

    CUDA_VISIBLE_DEVICES=$CVD timeout $TIME_LIMIT uv run python scripts/evaluate.py \
        --structures_path="$RESULTS_PATH/result_0/projected/generated_crystals_cif" \
        --relax=True --structure_matcher='disordered' \
        --save_as="$RESULTS_PATH/result_0/projected/metrics.json" > $RESULTS_PATH/result_0/projected/eval.log &
}

# Run for different parameters
# generate_and_evaluate gs ge ts te
generate_and_evaluate   1     1     0.95 0.0
generate_and_evaluate   0.1   0.1   0.95 0.0
generate_and_evaluate   0.01  0.01  0.95 0.0
generate_and_evaluate   0.001 0.001 0.95 0.0

generate_and_evaluate   0.01  1     0.95 0.0
generate_and_evaluate   1     0.01  0.95 0.0

# generate_and_evaluate   0.1 0.001 1 0.0
# generate_and_evaluate   0.01 0.001 1 0.0
# generate_and_evaluate   0.001 0.001 1 0.0
# generate_and_evaluate   0.001 0.01 1 0.0
# generate_and_evaluate   0.001 0.1 1 0.0

# generate_and_evaluate   0.01 0.0001 1 0.0
# generate_and_evaluate   0.001 0.0001 1 0.0
# generate_and_evaluate   0.0001 0.0001 1 0.0
# generate_and_evaluate   0.0001 0.001 1 0.0
# generate_and_evaluate   0.0001 0.01 1 0.0


echo "All jobs completed."
