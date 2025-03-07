# generate batch_size * num_batches samples
cd /home/holywater2/crystal_gen/mattergen

for spg in 6 38 123 139 166 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/spg_cond/spg${spg}_2000/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=True --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/metrics.json"
done


for spg in 6 38 123 139 166 216; do
    export RESULTS_PATH=/home/holywater2/crystal_gen/mattergen/results/spg_cond/spg${spg}_2000_gdf0/generated_crystals_cif
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/evaluate.py --structures_path=$RESULTS_PATH\
        --relax=True --structure_matcher='disordered'\
        --save_as="$RESULTS_PATH/metrics.json"
done