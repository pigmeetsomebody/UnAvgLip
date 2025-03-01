#!/bin/bash

# Function to extract video and audio sections
extract_sections() {
  input_video=$1
  base_name=$2
  # output_dir=$2
  # split=$3
  duration=$(ffmpeg -i "$input_video" 2>&1 | grep Duration | awk '{print $2}' | tr -d ,)
  IFS=: read -r hours minutes seconds <<< "$duration"

  #   # Save extracted audio

  output_audio="${output_dir}/audios/${base_name}.wav"
  mkdir -p "${output_dir}/audios"
  ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$output_audio"

# Create and update the config.yaml file
  echo "task_0:" > config.yaml
  echo "  video_path: \"$input_video\"" >> config.yaml
  echo "  audio_path: \"$output_audio\"" >> config.yaml
  echo "Run the Python script input_video: $input_video, audio path: $output_audio"
  # Run the Python script with the current config.yaml
  python -m scripts.data --inference_config config.yaml --folder_name "$base_name" --result_dir "$output_dir"

}

# Main script
if [ $# -lt 2 ]; then
  echo "Usage: $0 <output_directory> <input_directory>"
  exit 1
fi

output_dir=$2
input_dir=$1

# Initialize JSON arrays
train_json_array="["
test_json_array="["

# List all .mp4 files in the input directory recursively and shuffle them
all_videos=($(find "$input_dir" -type f -name "*.mp4" | shuf))

# Get the total number of videos
total_videos=${#all_videos[@]}
train_count=$((total_videos * 8 / 10))  # 80% for training

# Split into train and test sets
train_videos=("${all_videos[@]:0:$train_count}")
test_videos=("${all_videos[@]:$train_count}")

# Process training videos
for input_video in "${train_videos[@]}"; do
  base_name=$(echo "$input_video" | sed -E 's|.*/([a-zA-Z0-9_]+_[0-9]+)\.mp4|\1|')
  txt_file="${input_video%.mp4}.txt"

  echo "Processing train video: $input_video, base name: $base_name, txt_file: $txt_file"
  # Extract sections and run the Python script for each section
  extract_sections "$input_video" "$base_name"
  # extract_sections "$input_video" "$output_dir/train" "train"
  # rm $input_video
  # rm $txt_file

  # # Add entry to train JSON array
  # train_json_array+="\"../${output_dir}/images/$base_name\","
  # train_json_array="${train_json_array%,}]"
  # echo "$train_json_array" > train_lrw.json
  test_json_array+="\"${output_dir}/images/$base_name\","

done

# Process testing videos
for input_video in "${test_videos[@]}"; do
#   base_name=$(basename "$input_video" .mp4)
  # base_name=$(echo "$input_video" | sed -E 's|.*/(id[0-9]+)/(.*)/.*|\1_\2|')
  base_name=$(echo "$input_video" | sed -E 's|.*/([a-zA-Z0-9_]+_[0-9]+)\.mp4|\1|')


  echo "Processing train video: $input_video, base name: $base_name"

#   # Extract sections and run the Python script for each section
  extract_sections "$input_video" "$base_name"
  # rm $input_video
  # rm $txt_file  

  # Save extracted audio
#   output_audio="./new_train_data/audios/${base_name}.wav"
#   # mkdir -p "./new_train_data/audios"
#   ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$output_audio"

  # Add entry to test JSON array
  test_json_array+="\"${output_dir}/images/$base_name\","
done

# Remove trailing commas and close JSON arrays

# Remove trailing commas and close JSON arrays
train_json_array="${train_json_array%,}]"
test_json_array="${test_json_array%,}]"

# Write JSON arrays to the correct files
echo "$train_json_array" > train.json
echo "$test_json_array" > test.json


echo "Training and testing split complete."
