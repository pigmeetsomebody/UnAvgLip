export VAE_MODEL="../models/sd-vae-ft-mse/"
export DATASET="../new_train_data"
export UNET_CONFIG="../models/musetalk/musetalk.json"

accelerate launch adapter_linear_train.py \
--mixed_precision="fp16" \
--unet_config_file=$UNET_CONFIG \
--pretrained_model_name_or_path=$VAE_MODEL \
--data_root=$DATASET \
--train_batch_size=2 \
--gradient_accumulation_steps=1 \
--gradient_checkpointing \
--reconstruction \
--max_train_steps=2000 \
--learning_rate=1e-06 \
--max_grad_norm=1 \
--lr_scheduler="cosine" \
--lr_warmup_steps=0 \
--output_dir="linear23" \
--val_out_dir='linear23' \
--testing_speed \
--checkpointing_steps=500 \
--validation_steps=50 \
--resume_from_checkpoint="latest" \
--syncnet_checkpoint_path="./models/checkpoints/lipsync_expert.pth" \
--disc_checkpoint_path="./models/checkpoints/visual_quality_disc.pth" \
--use_audio_length_left=2 \
--use_audio_length_right=2 \
--n_face_id_cond=5 \
--whisper_model_type="tiny" \
--train_json="../train.json" \
--val_json="../test.json" \
--logging_dir="linear23"