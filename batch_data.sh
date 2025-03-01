#!/bin/bash

# Function to extract video and audio sections
extract_sections() {
  input_video=$1
  base_name=$(basename "$input_video" .mp4)
  output_dir=$2
  split=$3
  duration=$(ffmpeg -i "$input_video" 2>&1 | grep Duration | awk '{print $2}' | tr -d ,)
  IFS=: read -r hours minutes seconds <<< "$duration"
  total_seconds=$((10#${hours}*3600 + 10#${minutes}*60 + 10#${seconds%.*}))
  chunk_size=180  # 3 minutes in seconds
  index=0

  mkdir -p "$output_dir"

  while [ $((index * chunk_size)) -lt $total_seconds ]; do
    start_time=$((index * chunk_size))
    section_video="${output_dir}/${base_name}_part${index}.mp4"
    section_audio="${output_dir}/${base_name}_part${index}.wav"
    
    # Extract video and audio sections
    ffmpeg -i "$input_video" -ss "$start_time" -t "$chunk_size" -r 25 -c copy "$section_video"
    ffmpeg -i "$input_video" -ss "$start_time" -t "$chunk_size" -q:a 0 -ar 16000 -map a "$section_audio"
    
    # Create and update the config.yaml file
    echo "task_0:" > config.yaml
    echo "  video_path: \"$section_video\"" >> config.yaml
    echo "  audio_path: \"$section_audio\"" >> config.yaml

    # Run the Python script with the current config.yaml
    python -m scripts.data --inference_config config.yaml --folder_name "$base_name"
    
    index=$((index + 1))
  done
  # Clean up save folder
  rm -rf $output_dir
}

# Main script
if [ $# -lt 2 ]; then
  echo "Usage: $0 <output_directory> <input_directory>"
  exit 1
fi

output_dir=$1
input_dir=$2

# Initialize JSON arrays
train_json_array="["
test_json_array="["

# List all .mp4 files in the input directory and shuffle them
all_videos=($(ls "$input_dir"/*.mp4 | shuf))

# Get the total number of videos
total_videos=${#all_videos[@]}
train_count=$((total_videos * 8 / 10))  # 80% for training

# Split into train and test sets
train_videos=("${all_videos[@]:0:$train_count}")
test_videos=("${all_videos[@]:$train_count}")

# Process training videos
for input_video in "${train_videos[@]}"; do
  base_name=$(basename "$input_video" .mp4)
  echo "Processing train video: $input_video, base name: $base_name"
  # Extract sections and run the Python script for each section
  extract_sections "$input_video" "$output_dir/train" "train"

  # Save extracted audio
  output_audio="./new_train_data/audios/${base_name}.wav"
  # mkdir -p "./new_train_data/audios"
  ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$output_audio"

  # Add entry to train JSON array
  train_json_array+="\"../new_train_data/images/$base_name\","
done

# Process testing videos
for input_video in "${test_videos[@]}"; do
  base_name=$(basename "$input_video" .mp4)

  echo "Processing test video: $input_video, base name: $base_name"
  
  # Extract sections and run the Python script for each section
  extract_sections "$input_video" "$output_dir/test" "test"
  

  # Save extracted audio
  output_audio="./new_train_data/audios/${base_name}.wav"
  # mkdir -p "./new_train_data/audios"
  ffmpeg -i "$input_video" -q:a 0 -ar 16000 -map a "$output_audio"

  # Add entry to test JSON array
  test_json_array+="\"../new_train_data/images/$base_name\","
done

# Remove trailing commas and close JSON arrays
train_json_array="${train_json_array%,}]"
test_json_array="${test_json_array%,}]"

# Write JSON arrays to the correct files
echo "$train_json_array" > train.json
echo "$test_json_array" > test.json

echo "Training and testing split complete."
