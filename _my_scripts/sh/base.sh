export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH --batch_size=16 --num_batches 1 \
    --sampling_config_name default
