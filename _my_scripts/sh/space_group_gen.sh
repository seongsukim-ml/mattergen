export MODEL_NAME=space_group
export MODEL_PATH="checkpoints/$MODEL_NAME"  # Or provide your own model
export RESULTS_PATH="results/$MODEL_NAME/"  # Samples will be written to this directory, e.g., `results/dft_mag_density`

# Generate conditional samples with a target magnetic density of 0.15
python scripts/generate.py $RESULTS_PATH $MODEL_PATH --batch_size=2 \
    --checkpoint_epoch=last \
    --diffusion_guidance_factor=2.0 \
    --properties_to_condition_on={"space_group":1}
