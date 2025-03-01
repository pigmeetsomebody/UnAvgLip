filelist=$1
result_dir=$2
data_root=$3
rm eval_config.yaml
rm eval_config0.yaml
rm adapter_eval_result.yaml
rm musetalk_eval_result.yaml
rm wav2lip_eval_result.yaml
rm retalking_eval_result.yaml
rm talk_lip_g_eval_result.yaml
rm talk_lip_gc_eval_result.yaml
# rm sadtalker.yaml

adapter_path="${result_dir}/adapter"
musetalk_path="${result_dir}/musetalk"
wav2lip_path="${result_dir}/wav2lip"
sadtalker_path="${result_dir}/sadtalker"
talk_lip_g_path="/date/zhuyy/TalkLip/results/talklip_g"
talk_lip_gc_path="/date/zhuyy/TalkLip/results/talklip_gc"
video_retalking_path="${result_dir}/video_retalking"
wav2lip_checkpoint="/home/zhuyy/Wav2Lip/checkpoints/wav2lip.pth"


mkdir $adapter_path
mkdir $musetalk_path
mkdir $wav2lip_path
mkdir $video_retalking_path
source /home/zhuyy/anaconda3/etc/profile.d/conda.sh
while IFS= read -r line || [ -n "$line" ]; do
    echo "$line"
    filename=$(basename "$line")
    echo "Processing train video: $line, base name: $filename"
    cd ~/MuseTalk
    conda activate musetalk
    if [[ "$filename" == *.mp4 ]]; then
        base_name="${filename%.mp4}"
        input_audio="${result_dir}/${base_name}.wav"
        input_video="${data_root}/${line}"
        if [[ -e "$input_audio" ]]; then
            echo "文件 $input_audio 已经存在，跳过处理。"
        else
            echo "文件 $input_audio 不存在，处理。"
            ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$input_audio"
        fi
        # ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$input_audio"

        echo "task_$base_name:" > eval_config.yaml
            echo "  video_path: \"$input_video\"" >> eval_config.yaml
            echo "  audio_path: \"$input_audio\"" >> eval_config.yaml
            echo "  crop_path: \"\"" >> eval_config.yaml
            echo "  face_embs_path: \"\"" >> eval_config.yaml
            echo "  bbox_shift: 0" >> eval_config.yaml
            echo "  n_cond: 1" >> eval_config.yaml
            echo "  guide_scale: 0.25" >> eval_config.yaml
        

        echo "task_$base_name:" >> adapter_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> adapter_eval_result.yaml
            echo "  generated_video_path: ${adapter_path}/${base_name}_${base_name}/" >> adapter_eval_result.yaml
            echo "  result_save_path: ${adapter_path}" >> adapter_eval_result.yaml


        echo "task_$base_name:" > eval_config0.yaml
            echo "  video_path: \"$input_video\"" >> eval_config0.yaml
            echo "  audio_path: \"$input_audio\"" >> eval_config0.yaml
            echo "  crop_path: \"\"" >> eval_config0.yaml
            echo "  face_embs_path: \"\"" >> eval_config0.yaml
            echo "  bbox_shift: 0" >> eval_config0.yaml
            echo "  n_cond: 1" >> eval_config0.yaml
            echo "  guide_scale: 0" >> eval_config0.yaml


        cd ~/MuseTalk
        echo "task_$base_name:" >> musetalk_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> musetalk_eval_result.yaml
            echo "  generated_video_path: ${musetalk_path}/${base_name}_${base_name}/" >> musetalk_eval_result.yaml
            echo "  result_save_path: ${musetalk_path}" >> musetalk_eval_result.yaml
    
        # generate wav2lip
        cd ~/Wav2Lip
        python inference.py --checkpoint_path "$wav2lip_checkpoint" --face  "$input_video" --audio "$input_audio"
        mv ~/Wav2Lip/results/result_voice.mp4 "$wav2lip_path"/"$base_name"_"$base_name".mp4
        cd ~/MuseTalk
        echo "task_$base_name:" >> wav2lip_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> wav2lip_eval_result.yaml
            echo "  generated_video_path: ${wav2lip_path}/${base_name}_${base_name}.mp4" >> wav2lip_eval_result.yaml
            echo "  result_save_path: ${wav2lip_path}" >> wav2lip_eval_result.yaml


        # generate video-retalking
        conda init
        cd ~/video-retalking
        conda activate video_retalking
        python3 inference.py  --face "$input_video" --audio "$input_audio" --outfile "$video_retalking_path"/"$base_name"_"$base_name".mp4
        
        cd ~/MuseTalk
        echo "task_$base_name:" >> retalking_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> retalking_eval_result.yaml
            echo "  generated_video_path: ${video_retalking_path}/${base_name}_${base_name}.mp4" >> retalking_eval_result.yaml
            echo "  result_save_path: ${video_retalking_path}" >> retalking_eval_result.yaml


        echo "task_$base_name:" >> talk_lip_g_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> talk_lip_g_eval_result.yaml
            echo "  generated_video_path: ${talk_lip_g_path}/${base_name}.mp4" >> talk_lip_g_eval_result.yaml
            echo "  result_save_path: ${talk_lip_g_path}" >> talk_lip_g_eval_result.yaml

        echo "task_$base_name:" >> talk_lip_gc_eval_result.yaml
            echo "  gt_video_path: ${adapter_path}/${base_name}" >> talk_lip_gc_eval_result.yaml
            echo "  generated_video_path: ${talk_lip_gc_path}/${base_name}.mp4" >> talk_lip_gc_eval_result.yaml
            echo "  result_save_path: ${talk_lip_gc_path}" >> talk_lip_gc_eval_result.yaml
        # echo "task_$base_name:" >> sadtalker.yaml
        #     echo "  gt_video_path: ${adapter_path}/${base_name}" >> sadtalker.yaml
        #     echo "  generated_video_path: ${sadtalker_path}/${base_name}.mp4" >> sadtalker.yaml
        #     echo "  result_save_path: ${sadtalker_path}" >> sadtalker.yaml
        
    else
        echo "task_$filename:" > eval_config.yaml 
            echo "  video_path: ~/MuseTalk/new_train_data/images/$filename/" >> eval_config.yaml
            echo "  audio_path: ~/MuseTalk/new_train_data/audios/$filename.wav" >> eval_config.yaml
            echo "  crop_path: ~/MuseTalk/new_train_data/images/$filename/" >> eval_config.yaml
            echo "  face_embs_path: \"\"" >> eval_config.yaml
            echo "  bbox_shift: 0" >> eval_config.yaml
            echo "  n_cond: 1" >> eval_config.yaml
            echo "  guide_scale: 0.25" >> eval_config.yaml  
        python -m scripts.adapter_v_inference --inference_config eval_config.yaml  --result_dir "$adapter_path"
        echo "task_$filename:" > eval_config0.yaml 
            echo "  video_path: ~/MuseTalk/new_train_data/images/$filename/" >> eval_config.yaml
            echo "  audio_path: ~/MuseTalk/new_train_data/audios/$filename.wav" >> eval_config.yaml
            echo "  crop_path: ~/MuseTalk/new_train_data/images/$filename/" >> eval_config.yaml
            echo "  face_embs_path: \"\"" >> eval_config.yaml
            echo "  bbox_shift: 0" >> eval_config.yaml
            echo "  n_cond: 1" >> eval_config.yaml
            echo "  guide_scale: 0" >> eval_config.yaml  
        python -m scripts.adapter_v_inference --inference_config eval_config0.yaml  --result_dir "$musetalk_path"
    fi
done < "$filelist"

echo "results"
conda activate musetalk

python -m scripts.adapter_v_inference --inference_config eval_config.yaml  --result_dir "$adapter_path"
python -m scripts.adapter_v_inference --inference_config eval_config0.yaml  --result_dir "$musetalk_path"

python -m scripts.evaluate_batch --evaluate_config adapter_eval_result.yaml
python -m scripts.evaluate_batch --evaluate_config musetalk_eval_result.yaml
python -m scripts.evaluate_batch --evaluate_config retalking_eval_result.yaml
python -m scripts.evaluate_batch --evaluate_config wav2lip_eval_result.yaml

# python -m scripts.evaluate_batch --evaluate_config talk_lip_g_eval_result.yaml
# python -m scripts.evaluate_batch --evaluate_config talk_lip_gc_eval_result.yaml
# python -m scripts.evaluate_batch --evaluate_config sadtalker.yaml
