# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

for spg in 6 38 123 139 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/prim/spg${spg}_gs1_ge0.5_ts0.95_te0.0_prim_v2/result_0/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=True --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/metrics.json"
done


for spg in 6 38 123 139 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/prim/spg${spg}_gs1_ge0.5_ts0.95_te0.0_prim_v2/result_0/projected/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=True --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/metrics.json"
done


for spg in 6 38 123 139 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/prim/spg${spg}_gs1_ge0.5_ts0.95_te0.0_prim_v2/result_0/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=False --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/no_relax_metrics.json"
done


for spg in 6 38 123 139 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/prim/spg${spg}_gs1_ge0.5_ts0.95_te0.0_prim_v2/result_0/projected/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=False --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/no_relax_metrics.json"
done