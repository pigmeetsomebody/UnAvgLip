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
import random

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder, get_face_embedding
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
import numpy


# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:1")
timesteps = torch.tensor([0], device=device)

@torch.no_grad()
def main(args):
    global pe
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]
        bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
        args.crop_path = inference_config[task_id]["crop_path"]

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}"
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
        land_mark_save_path =  os.path.join(args.result_dir, input_basename+"_landmark.pkl")
        os.makedirs(result_img_save_path,exist_ok =True)
        result_crop_save_path = os.path.join(result_img_save_path, output_basename)
        os.makedirs(result_crop_save_path, exist_ok =True)
        
        if args.output_vid_name is None:
            output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
        else:
            output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
        ############################################## extract frames from source video ##############################################
        if get_file_type(video_path)=="video":
            save_dir_full = os.path.join(args.result_dir, input_basename)
            os.makedirs(save_dir_full,exist_ok = True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
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

        #print(input_img_list)
        ############################################## extract audio feature ##############################################
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
        ############################################## preprocess input image  ##############################################
        print(f"crop_coord_save_path: {crop_coord_save_path}")

        if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
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
                

        input_latent_list = []
        copy_flag = False
        crop_frames = []
        if not (args.crop_path == None) and os.path.exists(args.crop_path):
            croped_frame_list = glob.glob(os.path.join(args.crop_path, '*.[jpJP][pnPN]*[gG]'))
            croped_frame_list = sorted(croped_frame_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            crop_frames = read_imgs(croped_frame_list)
        else:
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                # crop_frame = frame
                crop_frames.append(crop_frame)
                # latents = vae.get_latents_for_unet(crop_frame)
                # input_latent_list.append(latents)
        
        for idx in range(0, len(crop_frames)):
            latents = vae.get_latents_for_unet(crop_frames[idx])
            input_latent_list.append(latents)


        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)
        batch_size = args.batch_size
        gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
        res_frame_list = []
        for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype) # torch, B, 5*N,384
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
            

                
        ############################################## pad to full image ##############################################
        print(f"pad talking image to original video: {len(res_frame_list)} frames")
        for i, res_pil in enumerate(tqdm(res_frame_list)):
            res_frame = cv2.cvtColor(numpy.asarray(res_pil),cv2.COLOR_RGB2BGR)  
            if (args.crop_path == None) or not os.path.exists(args.crop_path):
                bbox = coord_list_cycle[i%(len(coord_list_cycle))]
                ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                except:
                    print(bbox)
                    continue
                combine_frame = get_image(ori_frame,res_frame,bbox)
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
                cv2.imwrite(f"{result_crop_save_path}/{str(i).zfill(8)}.png", res_frame)
            else:
                cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", res_frame)

        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        # os.remove("temp.mp4")
        # shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test.yaml")
    parser.add_argument("--bbox_shift", type=int, default=15)
    parser.add_argument("--result_dir", default='./results', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        default=True,
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    parser.add_argument("--preprocessing",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )



    args = parser.parse_args()
    main(args)
