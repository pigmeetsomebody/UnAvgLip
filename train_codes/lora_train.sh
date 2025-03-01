export VAE_MODEL="../models/sd-vae-ft-mse/"
export DATASET="../new_train_data"
export UNET_CONFIG="../models/musetalk/musetalk.json"

accelerate launch adapter_lora_train.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=1 \
--reconstruction \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--max_train_steps=20000 \
--learning_rate=1e-06 \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="lora1227" \
--val_out_dir='lora1227' \
--testing_speed \
--checkpointing_steps=10000 \
--validation_steps=200 \
--resume_from_checkpoint="latest" \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--whisper_model_type="tiny" \
--train_json="../train.json" \
--val_json="../test.json" \
--logging_dir="lora1227"