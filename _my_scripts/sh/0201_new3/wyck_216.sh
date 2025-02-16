spg=216
CVD=7
vs="prim_mix"
gs=0.1
ge=1
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
    --sampling_config_name wyck_prim_new_mix \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\",\"sampler_partial.ts=$ts\",\"sampler_partial.te=$te\"]


CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/metrics.json"

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/projected/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/projected/metrics.json"


spg=216
CVD=7
gs=1
ge=0.1
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
    --sampling_config_name wyck_prim_new_mix \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\",\"sampler_partial.ts=$ts\",\"sampler_partial.te=$te\"]


CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/metrics.json"

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/projected/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/projected/metrics.json"

spg=216
CVD=7
gs=1
ge=1
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
    --sampling_config_name wyck_prim_new_mix \
    --sampling_config_overrides=[\"condition_loader_partial.space_groups=${spg}\",\"sampler_partial.guidance_start=$gs\",\"sampler_partial.guidance_end=$ge\",\"sampler_partial.ts=$ts\",\"sampler_partial.te=$te\"]


CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/metrics.json"

CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH"/result_0/projected/generated_crystals_cif" \
    --relax=True --structure_matcher='disordered'\
    --save_as="$RESULTS_PATH/result_0/projected/metrics.json"