spg=123
gs=0.01
ge=0.2
CVD=0
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg${spg}_gs${gs}_ge${ge}_v3  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=50 --num_batches 1 \
    --sampling_config_name wyck \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"]