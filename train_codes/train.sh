export VAE_MODEL="../models/sd-vae-ft-mse/"
export DATASET="../new_train_data"
export UNET_CONFIG="../models/musetalk/musetalk.json"

accelerate launch train_faceid_adapter.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=2 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--reconstruction \
--max_train_steps=10000 \
--learning_rate=1e-05 \
--max_grad_norm=3 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="adapterv0218" \
--val_out_dir='adapterv0218' \
--testing_speed \
--checkpointing_steps=5000 \
--validation_steps=100 \
--resume_from_checkpoint="latest" \
--syncnet_checkpoint_path="./models/checkpoints/lipsync_expert.pth" \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--whisper_model_type="tiny" \
--train_json="../train.json" \
--val_json="../test.json" \
--logging_dir="adapterv0218"