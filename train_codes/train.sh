export VAE_MODEL="../models/sd-vae-ft-mse/"
export DATASET="../new_train_data"
export UNET_CONFIG="../models/musetalk/musetalk.json"

accelerate launch train.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=256 \
--gradient_accumulation_steps=16 \
--gradient_checkpointing \
--reconstruction \
--max_train_steps=100000 \
--learning_rate=1e-06 \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="output" \
--val_out_dir='adapter_val' \
--testing_speed \
--checkpointing_steps=5000 \
--validation_steps=100 \
--resume_from_checkpoint="latest" \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--whisper_model_type="tiny" \
--train_json="../train.json" \
--val_json="../test.json" 