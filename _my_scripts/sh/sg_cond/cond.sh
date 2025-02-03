spg=6
CVD=6
gdf=0

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}

spg=123

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}

spg=166

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}

###############################################################################################
spg=216
CVD=7
gdf=0

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}

spg=38

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}

spg=139

export MODEL_PATH=checkpoints/space_group  # Or provide your own model
export RESULTS_PATH=results/spg_cond/spg${spg}_2000_gdf${gdf}  # Samples will be written to this directory
# export SAMPLING_CONF_PATH=/home/holywater2/crystal_gen/mattergen/sampling_conf/wyck.yaml

cd /home/holywater2/crystal_gen/mattergen

# generate batch_size * num_batches samples
CUDA_VISIBLE_DEVICES=$CVD uv run python scripts/generate.py $RESULTS_PATH $MODEL_PATH \
    --batch_size=100 --num_batches 20 \
    --sampling_config_name default \
    --properties_to_condition_on="{'space_group': ${spg}}" \
    --diffusion_guidance_factor=${gdf}