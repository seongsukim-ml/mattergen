spg=216
gs=1
ge=1
CVD=3
vs="prim_new"
bs=10
nb=10
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/prim/spg${spg}_gs${gs}_ge${ge}_${vs}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py \
    results/prim/spg${spg}_gs${gs}_ge${ge}_${vs}\
    $MODEL_PATH \
    --batch_size=$bs --num_batches $nb \
    --sampling_config_name wyck_prim_new \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\"] \
    --project_save True

export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/prim/spg6_gs1_ge1_prim_new2/result_0
uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/metrics.json"