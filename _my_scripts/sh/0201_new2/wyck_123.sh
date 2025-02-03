spg=123
CVD=5
vs="prim_v2"
gs=1
ge=0.5
ts=0.95
te=0.0
bs=20
nb=10
export MODEL_PATH=checkpoints/mattergen_base  # Or provide your own model
export RESULTS_PATH=results/prim/spg${spg}_gs${gs}_ge${ge}_ts${ts}_te${te}_${vs}  # Samples will be written to this directory

# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate_debug.py \
    results/prim/spg${spg}_gs${gs}_ge${ge}_ts${ts}_te${te}_${vs}\
    $MODEL_PATH \
    --project_save True \
    --batch_size=$bs --num_batches $nb \
    --sampling_config_name wyck_prim_new \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\
                                 \"sampler_partial.guidance_start=$gs\",\
                                 \"sampler_partial.guidance_end=$ge\",\
                                 \"sampler_partial.ts=$ts\",\
                                 \"sampler_partial.te=$te\"]

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/metrics.json"