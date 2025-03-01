import cv2
import os
# import dlib
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
import uuid

from musetalk.utils.utils import get_file_type,get_video_fps, search_audios_pair_images, codebook_gen, knn_codebook_gen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder, get_face_embedding, extract_audio_from_video
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
from train_codes.utils.audio import load_wav, melspectrogram, hp, save_wav
import shutil
import gc

# for extract face_id
import insightface
from  insightface.app import FaceAnalysis
# 3dmm extraction
import safetensors
import safetensors.torch 


import scipy.io as scio
from scipy.io import loadmat, savemat

import glob
from time import strftime
from PIL import Image
import random
import warnings



from scipy.io import savemat, loadmat

RESIZED_IMG = 256

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)


# def selected_random_face_embedings(frames, target_num, save_dir, vidname):
#     i = 0
#     face_embs = []
#     # for debug
#     image = Image.new('RGB', (RESIZED_IMG*target_num, RESIZED_IMG // 2))
#     while i < target_num:
#         random_frame = random.choice(frames)
#         height = random_frame.shape[1]
#         # Define the lip region as the lower half
#         y1, y2 = height // 2, height
#         cropped_lip_region = random_frame[y1:y2, :]
#         cropped_lip_image = Image.fromarray(cv2.cvtColor(cropped_lip_region, cv2.COLOR_BGR2RGB))
#         faces = face_app.get(cropped_lip_region)
#         if len(faces) <= 0:
#             print("no face detect!")
#             continue
#         else:
#             image.paste(cropped_lip_image, (RESIZED_IMG*i, 0))
#             image_path = os.path.join(save_dir, f"{vidname}.png")
#             image.save(image_path)
#             face_embs.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
#             i += 1
#     face_embeds_ts = torch.cat(face_embs, dim=1)
#     face_emb_path = os.path.join(save_dir, f"{vidname}.pt")
#     torch.save(face_embeds_ts, face_emb_path)
#     return face_emb_path

def find_minimal_mouth_diff(lms):
    """
    Find the index in coord_list where the difference between mouth inner points
    (66-67 and 62-63) is minimal.
    
    Args:
        coord_list: List of facial landmarks coordinates, where each element
                   contains the full set of facial landmarks
    
    Returns:
        int: Index of the coordinate set with minimal mouth difference
    """
    min_diff = float('inf')
    min_idx = -1
    
    for idx, landmarks in enumerate(lms):
        # Extract mouth inner landmarks
        range_plus = (landmarks[66]- landmarks[62])[1]
        
        # Update minimum if current difference is smaller
        if range_plus < min_diff:
            min_diff = range_plus
            min_idx = idx
    return min_idx


def get_largest_integer_filename(folder_path):
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        return -1

    # Get the list of files in the folder
    files = os.listdir(folder_path)

    # Check if the folder is empty
    if not files:
        return -1

    # Extract the integer part of filenames and find the largest
    largest_integer = -1
    for file in files:
        try:
            # Get the integer part of the filename
            file_int = int(os.path.splitext(file)[0])
            if file_int > largest_integer:
                largest_integer = file_int
        except ValueError:
            # Skip files that don't have an integer filename
            continue

    return largest_integer

def datagen(whisper_chunks,
            crop_images,
            batch_size=8,
            delay_frame=0):
    whisper_batch, crop_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(crop_images)
        crop_image = crop_images[idx]
        whisper_batch.append(w)
        crop_batch.append(crop_image)

        if len(crop_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            # latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, crop_batch
            whisper_batch, crop_batch = [], []

    # the last batch may smaller than batch size
    if len(crop_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        # latent_batch = torch.cat(latent_batch, dim=0)

        yield whisper_batch, crop_batch

@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    inference_config = OmegaConf.load(args.inference_config)
    total_audio_index=get_largest_integer_filename(f"{args.result_dir}/audios/{args.folder_name}")
    total_image_index=get_largest_integer_filename(f"{args.result_dir}/images/{args.folder_name}")
    temp_audio_index=total_audio_index
    temp_image_index=total_image_index
    print(f"resume from last total_audio_index: {total_audio_index}, total_image_index: {total_image_index}")
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]
        bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
        folder_name = args.folder_name
        if not os.path.exists(f"{args.result_dir}/images/{folder_name}/"):
            os.makedirs(f"{args.result_dir}/images/{folder_name}")
        if not os.path.exists(f"{args.result_dir}/audios/{folder_name}/"):
            os.makedirs(f"{args.result_dir}/audios/{folder_name}")
        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        npy_save_path = os.path.join(os.path.join(args.result_dir, "audios"), input_basename + ".npy")
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
        land_mark_save_path = os.path.join(result_img_save_path, input_basename+"landmark.pkl")
        os.makedirs(result_img_save_path,exist_ok =True)
        
        if args.output_vid_name is None:
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {result_img_save_path}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(result_img_save_path, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        elif get_file_type(video_path)=="image":
            input_img_list = [video_path, ]
            fps = args.fps
        elif os.path.isdir(video_path):  # input img folder
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = args.fps
        else:
            raise ValueError(f"{video_path} should be a video file, an image file or a directory of images")
        ############################################## extract audio feature ##############################################
        whisper_feature = audio_processor.audio2feat(audio_path)
        
        # ori_wav = load_wav(audio_path, hp.sample_rate)
        # ori_mels = melspectrogram(ori_wav).T
        for __ in range(0, len(whisper_feature) - 1, 2):  # -1 to avoid index error if the list has an odd number of elements
            # Combine two consecutive chunks
            concatenated_chunks = np.concatenate([whisper_feature[__], whisper_feature[__+1]], axis=0)
            # Save the pair to a .npy file
            np.save(f'{args.result_dir}/audios/{folder_name}/{total_audio_index+(__//2)+1}.npy', concatenated_chunks)
            temp_audio_index=(__//2)+total_audio_index+1
        total_audio_index=temp_audio_index
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)

        ############################################## preprocess input image  ##############################################
        gc.collect()
        if os.path.exists(crop_coord_save_path) and os.path.exists(land_mark_save_path):
            print("using extracted coordinates")
            with open(crop_coord_save_path,'rb') as f:
                coord_list = pickle.load(f)
            with open(land_mark_save_path, 'rb') as f:
                face_land_marks = pickle.load(f)
            frame_list = read_imgs(input_img_list)
        else:
            print("extracting landmarks...time consuming")
            coord_list, frame_list, face_land_marks = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(crop_coord_save_path, 'wb') as f:
                pickle.dump(coord_list, f)
            with open(land_mark_save_path, 'wb') as f:
                pickle.dump(face_land_marks, f)


                
        i = 0
        input_latent_list = []
        crop_i=0
        crop_data=[]
        pil_data = []

        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox

            x1=max(0,x1)
            y1=max(0,y1)
            x2=max(0,x2)
            y2=max(0,y2)

            if ((y2-y1)<=0) or ((x2-x1)<=0):
                continue
            crop_frame = frame[y1:y2, x1:x2]
            
            crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
            # latents = vae.get_latents_for_unet(crop_frame)
            crop_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
            crop_data.append(crop_frame)
            pil_data.append(crop_pil)
            # input_latent_list.append(latents)
            crop_i+=1
        
        # to smooth the first and the last frame
        # frame_list_cycle = frame_list + frame_list[::-1]
        # coord_list_cycle = coord_list + coord_list[::-1]
        # input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        crop_data = crop_data + crop_data[::-1]
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,crop_data, batch_size)
        crop_index=0
        for i, (whisper_batch,crop_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            for image, audio in zip(crop_batch, whisper_batch):
                temp_image_index = crop_index + total_image_index + 1
                print(f"save images in {args.result_dir}/images/{folder_name}")
                cv2.imwrite(f"{args.result_dir}/images/{folder_name}/{str(temp_image_index)}.png",image)
                crop_index+=1
        total_image_index=temp_image_index
        total_audio_index = temp_audio_index
        gc.collect()
        shutil.rmtree(result_img_save_path)

           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test_img.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--folder_name", default=f'{uuid.uuid4()}', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )

    args = parser.parse_args()
    main(args)


def process_audio(audio_path):
    whisper_feature = audio_processor.audio2feat(audio_path)
    np.save('audio/your_filename.npy', whisper_feature)

def mask_face(image):
    # Load dlib's face detector and the landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor_path = "/content/shape_predictor_68_face_landmarks.dat"  # Set path to your downloaded predictor file
    predictor = dlib.shape_predictor(predictor_path)

    # Load your input image
    # image_path = "/content/ori_frame_00000077.png"  # Replace with the path to your input image
    # image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detector(gray)

    # Process each detected face
    for face in faces:
        # Predict landmarks
        landmarks = predictor(gray, face)

        # The indices of nose landmarks are 27 to 35
        nose_tip = landmarks.part(33).y

        # Blacken the region below the nose tip
        blacken_area = image[nose_tip:, :]
        blacken_area[:] = (0, 0, 0)

    # Save the final image or display it
    # cv2.imwrite("output_image.jpg", image)
    return image