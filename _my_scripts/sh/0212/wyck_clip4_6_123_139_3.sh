#!/bin/bash

# Set number of threads
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

# General configurations
export CVD=6
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export VS="mask_0211_0"
export SPG=6
export BS=40
export NB=10
export TIME_LIMIT=14400
export CONFIG_NAME="wyck_prim_mask"

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
    # CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py \
    #     $RESULTS_PATH \
    #     $MODEL_PATH \
    #     --project_save True \
    #     --batch_size=$BS --num_batches $NB \
    #     --sampling_config_name $CONFIG_NAME \
    #     --sampling_config_overrides="[\"condition_loader_partial.space_groups=${SPG}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\",\"sampler_partial.ts=$ts\",\"sampler_partial.te=$te\"]" \
    # | tee $RESULTS_PATH/gen.log
    # mv $RESULTS_PATH/gen.log $RESULTS_PATH/result_0/gen.log

    echo "Running evaluation..."

    # Evaluate generated structures
    CUDA_VISIBLE_DEVICES=$CVD timeout $TIME_LIMIT uv run python scripts/evaluate.py \
        --structures_path="$RESULTS_PATH/result_0/generated_crystals_cif" \
        --relax=True --structure_matcher='disordered' \
        --save_as="$RESULTS_PATH/result_0/metrics.json" > $RESULTS_PATH/result_0/eval.log &
 
    CUDA_VISIBLE_DEVICES=$CVD timeout $TIME_LIMIT uv run python scripts/evaluate.py \
        --structures_path="$RESULTS_PATH/result_0/generated_crystals_cif_spg" \
        --relax=True --structure_matcher='disordered' \
        --save_as="$RESULTS_PATH/result_0/metrics_spg.json" > $RESULTS_PATH/result_0/eval.log &

    # CUDA_VISIBLE_DEVICES=$CVD timeout $TIME_LIMIT uv run python scripts/evaluate.py \
    #     --structures_path="$RESULTS_PATH/result_0/projected/generated_crystals_cif" \
    #     --relax=True --structure_matcher='disordered' \
    #     --save_as="$RESULTS_PATH/result_0/projected/metrics.json" > $RESULTS_PATH/result_0/projected/eval.log &
}

# Run for different parameters
# generate_and_evaluate gs ge ts te
generate_and_evaluate   1.000  1.000   1   0.0

#!/bin/bash

# General configurations
export CVD=7
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export VS="mask_0211_0"
export SPG=123
export BS=40
export NB=10
export CONFIG_NAME="wyck_prim_mask"

# Run for different parameters
# generate_and_evaluate gs ge ts te
generate_and_evaluate   1.000  1.000   1   0.0

# General configurations
export CVD=5
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export VS="mask_0211_0"
export SPG=139
export BS=40
export NB=10
export CONFIG_NAME="wyck_prim_mask"

# Run for different parameters
# generate_and_evaluate gs ge ts te
generate_and_evaluate   1.000  1.000   1   0.0