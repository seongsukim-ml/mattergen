spg=123
gs=0.1
ge=0.3
CVD=7
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg${spg}_gs${gs}_ge${ge}_v6  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 10 \
    --sampling_config_name wyck \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"]


spg=123
gs=1
ge=3
CVD=6
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg${spg}_gs${gs}_ge${ge}_v6  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 10 \
    --sampling_config_name wyck \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"]

spg=123
gs=5
ge=15
CVD=5
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg${spg}_gs${gs}_ge${ge}_v6  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 10 \
    --sampling_config_name wyck \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"]

spg=123
gs=2
ge=6
CVD=4
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/spg${spg}_gs${gs}_ge${ge}_v7  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 1 \
    --sampling_config_name wyck \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"]