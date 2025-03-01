import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from time import strftime
import zipfile

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil


# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)
processed_data_nums = 0
def unzip_all_zip_files(root_dir):
    # Walk through all directories and files recursively
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_file_path = os.path.join(root, file)
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        print(f"Extracting {zip_file_path}...")
                        zip_ref.extractall(root)
                        print(f"Extracted {zip_file_path} successfully.")
                except zipfile.BadZipFile as e:
                    print(f"Error extracting {zip_file_path}: {e}")
                except Exception as e:
                    print(f"An error occurred with {zip_file_path}: {e}")

def process_data(args, processed_data_nums = 0):
    video_dir_name = args.video_dir_name
    audio_dir_name = args.audio_dir_name
    data_image_dir_name = os.path.join(args.data_dir_name, 'images')
    train_filelists_name = args.train_filelists_name
    val_filelists_name = args.val_filelists_name
    if not os.path.exists(data_image_dir_name):
        os.makedirs(data_image_dir_name)
    data_audio_dir_name = os.path.join(args.data_dir_name, 'audios')
    if not os.path.exists(data_audio_dir_name):
        os.makedirs(data_audio_dir_name)
    for dirpath, dirnames, filenames in os.walk(video_dir_name):
        for filename in filenames:
            audio_file_name = filename.replace('.mp4', '.wav')
            audio_dir_path = dirpath.replace('train_video', 'train_audio')
            audio_path = os.path.join(audio_dir_path, audio_file_name)
            if filename.endswith('.mp4') and os.path.exists(audio_path):
                input_basename = os.path.basename(dirpath).split('.')[0]
                input_basename += filename.split('.')[0]
                video_path = os.path.join(dirpath, filename)
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                save_imgs_path = os.path.join(data_image_dir_name, input_basename)
                if not os.path.exists(save_imgs_path):
                    os.makedirs(save_imgs_path)
                print(f"store the images in {save_imgs_path}")
                cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_imgs_path}/%d.png"
                os.system(cmd)
                input_img_list = sorted(glob.glob(os.path.join(save_imgs_path, '*.[jpJP][pnPN]*[gG]')))
                print("extracting landmarks...time consuming")
                coord_list, frame_list = get_landmark_and_bbox(input_img_list)
                crop_i = 0
                for bbox, fram in zip(coord_list, frame_list):
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = max(0, x2)
                    y2 = max(0, y2)
                    if ((y2-y1) <= 0 or (x2-x1) <= 0):
                        print("wrong rector")
                        continue
                    crop_frame = fram[y1:y2, x1:x2]
                    crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(f"{save_imgs_path}/{str(crop_i)}.png", crop_frame)
                    crop_i += 1


                if processed_data_nums % 10 == 0:
                    with open(val_filelists_name, 'a') as f:
                        f.write(f"{input_basename} {len(input_img_list)}\n")
                else:
                    with open(train_filelists_name, 'a') as f:
                        f.write(f"{input_basename} {len(input_img_list)}\n")
                # 处理音频
                print(f"process audion: {audio_path} fps: {fps}...")
                whisper_feature = audio_processor.audio2feat(audio_path)
                whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
                print(f"processed whisper_feature: {len(whisper_chunks)}")

                for index, chunk in enumerate(whisper_chunks):
                    save_audios_path = os.path.join(data_audio_dir_name, input_basename)
                    if not os.path.exists(save_audios_path):
                        os.makedirs(save_audios_path)
                    save_audios_path = os.path.join(save_audios_path, f"{index}.npy")
                    np.save(save_audios_path, chunk)

                processed_data_nums += 1
                print(f"======processed data: {processed_data_nums}===============")




def main(args):
    video_dir_name = args.video_dir_name
    print(f"unzip all the video zip files in: {video_dir_name}...")
    # unzip_all_zip_files(video_dir_name)
    audio_dir_name = args.audio_dir_name
    print(f"unzip all the audio zip files in: {audio_dir_name}...")
    # unzip_all_zip_files(audio_dir_name)
    data_image_dir_name = os.path.join(args.data_dir_name, 'images')
    if not os.path.exists(data_image_dir_name):
        os.makedirs(data_image_dir_name)

    # train_filelists_name = args.train_filelists_name
    process_data(args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir_name", type=str, default="./data/train_video")
    parser.add_argument("--audio_dir_name", type=str, default="./data/train_audio")
    parser.add_argument("--data_dir_name", type=str, default="./data")
    parser.add_argument("--train_filelists_name", type=str, default="./train_codes/filelists/train.txt")
    parser.add_argument("--val_filelists_name", type=str, default="./train_codes/filelists/val.txt")
    args = parser.parse_args()
    print(args)
    main(args)
