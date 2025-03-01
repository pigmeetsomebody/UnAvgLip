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
from train_codes.utils.utils import prepare_mask_and_masked_image
from musetalk.utils.utils import get_file_type,get_video_fps, datagen_with_faceid, search_audios_pair_images, codebook_gen, knn_codebook_gen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder, get_face_embedding, extract_audio_from_video
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model
import shutil
import insightface
from  insightface.app import FaceAnalysis
from train_codes.ip_adapter.ip_adapter_faceid import MLPProjModel
from train_codes.ip_adapter.utils import is_torch2_available, register_cross_attention_hook, get_net_attn_map, attnmaps2images, concat_images
from train_codes.ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from train_codes.ip_adapter.ip_adapter import IPAdapterModule
from train_codes.ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans

from PIL import Image
import matplotlib.pyplot as plt

from accelerate import Accelerator
from scipy.spatial.distance import cosine



# 3dmm extraction
import safetensors
import safetensors.torch 
from face3dmm.face3d.util.preprocess import align_img
from face3dmm.face3d.util.load_mats import load_lm3d
from face3dmm.face3d.models import networks

import scipy.io as scio
from scipy.io import loadmat, savemat

from face3dmm.utils.croper import Preprocesser
import glob
from time import strftime

import warnings

from face3dmm.utils.safetensor_helper import load_x_from_safetensor 
from face3dmm.face3d.extract_kp_videos_safe import KeypointExtractor
from face3dmm.utils.init_path import init_path
from face3dmm.face3d.util.generate_facerender_batch import get_facerender_data_simple
from face3dmm.facerender.animate import AnimateFromCoeff
from face3dmm.preprocessing import preprocessing_images


from scipy.io import savemat, loadmat

RESIZED_IMG = 256
face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG // 2))

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def find_k_top_distinct_mounth_diff(lms, k=5, window_size=5):
    results = []
    if len(lms) == 0:
        return results
    first_lm = lms[0]
    min_diff = (first_lm[66]- first_lm[62])[1]
    max_diff = min_diff
    results.append(0)
    threshhod = 0.5
    while (len(results) < k):
        for idx in range(0, len(lms), window_size):
            # print(idx, lms[idx])
            lm = lms[idx]
            lm_range = (lm[66]- lm[62])[1]
            if lm_range >= min_diff and lm_range <= max_diff:
                continue
            lip_lm = lm[48:66]
            if lm_range < min_diff:
                cmp_idx = results[0]
                cmp_lip_lm = lms[cmp_idx][48:66]
                lip_lm_flat = lip_lm.flatten()
                cmp_lip_lm_flat = cmp_lip_lm.flatten()
                similarity = cosine_similarity(lip_lm_flat, cmp_lip_lm_flat)
                if similarity < threshhod:
                    results.insert(0, idx)
                    min_diff = lm_range
            elif lm_range > max_diff:
                cmp_idx = results[-1]
                cmp_lip_lm = lms[cmp_idx][48:66]
                # print(f"cmp_lip_lm: {cmp_lip_lm} \n lip_lm:{lip_lm}")
                # similarity = 1 - cosine(lip_lm, cmp_lip_lm)  # 余弦相似度 (1 - cosine 距离)
                lip_lm_flat = lip_lm.flatten()
                cmp_lip_lm_flat = cmp_lip_lm.flatten()
                similarity = cosine_similarity(lip_lm_flat, cmp_lip_lm_flat)
                if similarity < threshhod:
                    results.append(idx)
                    max_diff = lm_range
        threshhod += 0.1
    return results

def selected_upgrade_face_embeddings(frames, lms, target_num, save_dir, vidname):
    face_embs = []
    most_distinct_mouth_diff_indexs = find_k_top_distinct_mounth_diff(lms, target_num)
    print(f"most_distinct_mouth_diff_indexs: {most_distinct_mouth_diff_indexs}")
    for i, idx in enumerate(most_distinct_mouth_diff_indexs):
        target_frame = frames[idx]
        height = target_frame.shape[1]
        y1, y2 = height // 2, height
        cropped_lip_region = target_frame[y1:y2, :]
        faces = face_app.get(cropped_lip_region)
        if len(faces) <= 0:
            print("no face detect!")
            continue
        print(f"faces[0].normed_embedding.shape: {faces[0].normed_embedding.shape}")
        face_embs.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
    face_embeds_ts = torch.cat(face_embs, dim=1)
    print(f"face_embeds_ts.shape: {face_embeds_ts.shape}")
    face_emb_path = os.path.join(save_dir, f"{vidname}.pt")
    torch.save(face_embeds_ts, face_emb_path)
    return face_emb_path


def selected_random_face_embedings(frames, target_num, save_dir, vidname):
    i = 0
    face_embs = []
    # for debug
    image = Image.new('RGB', (RESIZED_IMG*target_num, RESIZED_IMG // 2))
    while i < target_num:
        random_frame = random.choice(frames)
        height = random_frame.shape[1]
        # Define the lip region as the lower half
        y1, y2 = height // 2, height
        cropped_lip_region = random_frame[y1:y2, :]
        cropped_lip_image = Image.fromarray(cv2.cvtColor(cropped_lip_region, cv2.COLOR_BGR2RGB))
        faces = face_app.get(cropped_lip_region)
        if len(faces) <= 0:
            print("no face detect!")
            continue
        else:
            image.paste(cropped_lip_image, (RESIZED_IMG*i, 0))
            image_path = os.path.join(save_dir, f"{vidname}.png")
            image.save(image_path)
            face_embs.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
            i += 1
    face_embeds_ts = torch.cat(face_embs, dim=1)
    face_emb_path = os.path.join(save_dir, f"{vidname}.pt")
    torch.save(face_embeds_ts, face_emb_path)
    return face_emb_path

accelerator = Accelerator(
    mixed_precision="fp16",
)

RESIZED_IMG = 256

# load model weights
audio_processor, vae, unet, pe = load_all_model()

# load adapter module
image_proj_model = MLPProjModel(
    cross_attention_dim=unet.model.config.cross_attention_dim,
    id_embeddings_dim=512,
    num_tokens=4,
)



# init adapter modules
# lora_rank = 128
# attn_procs = {}
# unet_sd = unet.model.state_dict()
# for name in unet.model.attn_processors.keys():
#     cross_attention_dim = None if name.endswith("attn1.processor") else unet.model.config.cross_attention_dim
#     if name.startswith("mid_block"):
#         hidden_size = unet.model.config.block_out_channels[-1]
#         print(f"mid_block hidden_size: {hidden_size}")
#     elif name.startswith("up_blocks"):
#         block_id = int(name[len("up_blocks.")])
#         hidden_size = list(reversed(unet.model.config.block_out_channels))[block_id]
#         print(f"up_blocks hidden_size: {hidden_size}")
#     elif name.startswith("down_blocks"):
#         block_id = int(name[len("down_blocks.")])
#         hidden_size = unet.model.config.block_out_channels[block_id]
#         print(f"down_blocks hidden_size: {hidden_size}")
#     if cross_attention_dim is None:
#         attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
#     else:
#         layer_name = name.split(".processor")[0]
#         weights = {
#             "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
#             "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
#         }
#         attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
#         attn_procs[name].load_state_dict(weights, strict=False)
# print(f"set unet atten processor: {attn_procs}")
# unet.model.set_attn_processor(attn_procs)
# adapter_modules = torch.nn.ModuleList(unet.model.attn_processors.values())




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ip_adapter = IPAdapterModule(unet.model, image_proj_model, adapter_modules)

ip_adapter = IPAdapterFaceID(unet.model, "train_codes/output/adapter0104/checkpoint-20000/pytorch_model.bin", accelerator.device, 4, 1, torch.float32)
ip_adapter.load_ip_adapter()
print(f"ip_adapter.attn_processors.values(): {unet.model.attn_processors.values()}")
ip_adapter = accelerator.prepare(
    ip_adapter,
)
#device = torch.device("cuda:1")
timesteps = torch.tensor([0], device=device)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
face_embedding_extractor = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_embedding_extractor.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG))



        


    




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

@torch.no_grad()
def main(args):
    global pe
    
    # if not (args.pretrained_ip_adapter_path == None):
    #     print("unet ckpt loaded")
    #     accelerator.load_state(args.pretrained_ip_adapter_path, strict=False)
        
    if args.use_float16 is True:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    
    
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    ip_adapter.set_scale(args.guide_scale)
    # ip_adapter.load_ip_adapter()
    for task_id in inference_config:
        video_path = inference_config[task_id]["video_path"]
        audio_path = inference_config[task_id]["audio_path"]
        bbox_shift = inference_config[task_id].get("bbox_shift", args.bbox_shift)
        args.face_embs_path = inference_config[task_id]["face_embs_path"]
        args.crop_path = inference_config[task_id]["crop_path"]
        

        input_basename = os.path.basename(video_path).split('.')[0]
        audio_basename  = os.path.basename(audio_path).split('.')[0]
        output_basename = f"{input_basename}_{audio_basename}_adapter_{args.guide_scale}_selected"
        
        result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
        atten_map_img_save_path = os.path.join(result_img_save_path, "atten_map_imgs")

        
        
        crop_coord_save_path = os.path.join(args.result_dir, input_basename+".pkl") # only related to video input
        land_mark_save_path =  os.path.join(args.result_dir, input_basename+"_landmark.pkl")
        os.makedirs(result_img_save_path,exist_ok =True)
        os.makedirs(atten_map_img_save_path, exist_ok=True)

        best_ref_save_path = os.path.join(result_img_save_path, 'best_ref_image.png')

        ori_image_save_dir = os.path.join(result_img_save_path, output_basename)
        os.makedirs(ori_image_save_dir,exist_ok =True)
        
        face3dmm_save_dir = os.path.join(result_img_save_path, 'face3dmm')
        os.makedirs(face3dmm_save_dir,exist_ok =True)

        if args.output_vid_name is None:
            output_vid_name = os.path.join(args.result_dir, output_basename + '_adapter' + "_" + str(args.preprocessing) + "_" + str(args.guide_scale) + ".mp4")
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
        use_saved_coord = args.use_saved_coord and (args.preprocessing is not True or (args.preprocessing is  True  and os.path.exists(best_ref_save_path)))
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
        # minimal_mouth_diff_index = find_minimal_mouth_diff(face_land_marks)
        # coord_list, frame_list, face_land_marks = get_landmark_and_bbox(input_img_list, bbox_shift)
        # k = 5
        # minimal_mouth_diff_indexs = find_k_top_distinct_mounth_diff(face_land_marks, k)
        # image = Image.new('RGB', (RESIZED_IMG*k, RESIZED_IMG))
        # for i, idx in enumerate(minimal_mouth_diff_indexs):
        #     pil_image = Image.fromarray(cv2.cvtColor(frame_list[idx], cv2.COLOR_BGR2RGB))
        #     image.paste(pil_image, (RESIZED_IMG*i, 0))
        # selected_imgs_path = os.path.join(args.result_dir, input_basename+"_selected_most_distinct_k.png")
        # image.save(selected_imgs_path)
        # print(f"minimal_mouth_diff_indexs: {minimal_mouth_diff_indexs}\n save_path: {selected_imgs_path}")
        # os._exit(0)
        

        i = 0
        input_latent_list = []
        copy_flag = False
        crop_frames = []
        faces_embeds = []
        if not (args.crop_path == None) and os.path.exists(args.crop_path):
            print(f"using cropped images: {args.crop_path}")
            crop_img_list = glob.glob(os.path.join(args.crop_path, '*.[jpJP][pnPN]*[gG]'))
            crop_img_list = sorted(crop_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            crop_frames = read_imgs(crop_img_list)
        else: 
            idx = 0
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                    # 提取人脸特征

                save_path = os.path.join(ori_image_save_dir, f"{idx}.png")
                cv2.imwrite(save_path, crop_frame)
                crop_frames.append(crop_frame)
                idx += 1
            # faces = face_embedding_extractor.get(crop_frame)
            # faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            # crop_frames.append(crop_frame)
            

            # faces_embeds.append(faceid_embeds)
            
            # latents = vae.get_latents_for_unet(crop_frame)
            # input_latent_list.append(latents)

        if not (args.face_embs_path == None) and os.path.exists(args.face_embs_path):
            faceid_embeds = torch.load(args.face_embs_path)
        else:

            face_embs_path = os.path.join(args.result_dir, input_basename+".pt")
            print(f"face_embs_path stored in {face_embs_path}")
            # if not os.path.exists(face_embs_path):
            #     face_embs_path = selected_random_face_embedings(frames=crop_frames, target_num=5, save_dir=args.result_dir, vidname=input_basename)
            # else:
            #     print(f"using existing landmarks: {face_embs_path}")
            # face_embs_path = selected_random_face_embedings(frames=crop_frames,lms=face_land_marks, target_num=5, save_dir=args.result_dir, vidname=input_basename)
            face_embs_path = selected_random_face_embedings(frames=crop_frames, target_num=5, save_dir=args.result_dir, vidname=input_basename)
            if os.path.exists(face_embs_path):
                faceid_embeds = torch.load(face_embs_path)
            else:
                faceid_embeds = torch.zeros(1, 5, 512)
        print(f"faceid_embeds.shape: {faceid_embeds.shape}")
        selected_idx = torch.randint(5, (1,)) 
        selected_face_emb = faceid_embeds[0, selected_idx, :]  # shape [1, 512]
        # selected_face_emb = selected_face_emb.unsqueeze(0)
        print(f"selected_face_emb.shape: {selected_face_emb.shape}")


        if args.preprocessing is True:
            if len(crop_frames) == 0:
                print('No face is detected in the input file')
            # if not use_saved_coord:
            #     minimal_mouth_diff_index = find_minimal_mouth_diff(face_land_marks)
            #     best_ref_img = ori_frames[minimal_mouth_diff_index]
            #     cv2.imwrite(best_ref_save_path, best_ref_img)
            #     print(f"get best_ref_img from face_land_marks, minimal_mouth_diff_index: {minimal_mouth_diff_index}")
            # else:
            #     if os.path.exists(best_ref_save_path):
            #         best_ref_img = Image.open(best_ref_save_path)
            #         print(f"get best_ref_img from best_ref_save_path, best_ref_save_path: {best_ref_save_path}")
            #     else:
            #         best_ref_img = ori_frames[0]
            #         print(f"get best_ref_img with default ori_frames[0]")
            crop_frames_dir = os.path.join(face3dmm_save_dir, 'images')
            if not os.path.exists(crop_frames_dir):
                input_img_list = glob.glob(os.path.join(ori_image_save_dir, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                ori_frames = [Image.open(image_path) for image_path in input_img_list]
                minimal_mouth_diff_index = find_minimal_mouth_diff(face_land_marks)
                best_ref_img = ori_frames[minimal_mouth_diff_index]
                crop_frames_dir = preprocessing_images(best_ref_img, ori_frames, face3dmm_save_dir, input_basename, device)
            print(f"crop_frames_dir: {crop_frames_dir}")
            crop_frames_list = glob.glob(os.path.join(crop_frames_dir, '*.[jpJP][pnPN]*[gG]'))
            crop_frames_list = sorted(crop_frames_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            crop_frames = read_imgs(crop_frames_list)
            
        input_img_list = glob.glob(os.path.join(ori_image_save_dir, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        ori_frames = read_imgs(input_img_list)
        for idx in range(0, len(crop_frames)):
            # if args.preprocessing is True:
            #     # ori_image = cv2.cvtColor(
            # np.array(ori_frames[idx]), cv2.COLOR_RGB2BGR)
            #     print(f"crop_frame: {crop_frames[idx].shape}, ori_frame:{ori_frames[idx].shape}")
            #     latents = vae.get_processed_latents_for_unet(crop_frames[idx], ori_frames[idx])
            # else:
            #     latents = vae.get_latents_for_unet(crop_frames[idx])
            latents = vae.get_latents_for_unet(crop_frames[idx])
            input_latent_list.append(latents)
        
        # while True:
        #     random_frame = random.choice(ori_frames)
        #     # random_frame = crop_frames[j]
        #     faces = face_embedding_extractor.get(random_frame)
        #     # print(f"detect faces: {len(faces)}")
        #     # j+=1
        #     if len(faces) <= 0:
        #         continue
        #     else:
        #         faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        #         faces_embeds = [faceid_embed] * len(input_latent_list)
        #         break
            
        # to smooth the first and the last frame
        frame_list_cycle = frame_list + frame_list[::-1]
        crop_image_cycle = crop_frames + crop_frames[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        faces_embeds_list = [selected_face_emb] * len(input_latent_list)
        faces_embeds_list_cycle = faces_embeds_list + faces_embeds_list[::-1]
        ############################################## inference batch by batch ##############################################
        print("start inference")
        video_num = len(whisper_chunks)

        best_matched_images = []
        # if get_file_type(video_path)=="video":
        #     reference_audio_path = os.path.join(result_img_save_path, input_basename+'.wav')
        #     if not os.path.exists(reference_audio_path):
        #         print(f'extract audio from video and save to {reference_audio_path}')
        #         extract_audio_from_video(video_path, reference_audio_path)
        #     if os.path.exists(reference_audio_path):
        #         print(f'extract audio from video and save to {reference_audio_path} succeed')
        #         print("start building codebook.")
        #         reference_whisper_feature = audio_processor.audio2feat(reference_audio_path)
        #         reference_whisper_chunks = audio_processor.feature2chunks(feature_array=reference_whisper_feature,fps=fps)
        #         n_clusters = min(3600, len(reference_whisper_chunks))
        #         knn = KNeighborsClassifier(n_neighbors=5)
        #         # knn = knn.to(vae.device)
        #         kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42)
        #         knn_codebook_gen(kmeans, knn, reference_whisper_chunks, input_latent_list, delay_frame=0)
        #         best_matched_image_save_dir = os.path.join(result_img_save_path, f'best_matched_images/')
        #         best_matched_images, faces_embeds = search_audios_pair_images(kmeans, knn, whisper_chunks, crop_frames, best_matched_image_save_dir)

        # faces_embeds_list_cycle = faces_embeds + faces_embeds[::-1]
        batch_size = args.batch_size
        # assert(len(faces_embeds) == len(whisper_chunks))
        gen = datagen_with_faceid(whisper_chunks,input_latent_list_cycle, faces_embeds_list_cycle, best_matched_images, crop_image_cycle, batch_size)
        res_frame_list = []
        if len(crop_frames) > 1:   
            crop_frames = random.sample(crop_frames, batch_size)
        print(f"crop_frames.shape: {len(crop_frames)}")
        for i, (whisper_batch,latent_batch, faces_embeds_batch, best_matched_images_batch, crop_images_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
            print(f"whisper_batch.shape:{whisper_batch.shape}, latent_batch: {latent_batch.shape}, faces_embeds_batch: {faces_embeds_batch.shape}")
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                         dtype=unet.model.dtype) # torch, B, 5*N,384
            
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            faces_embeds_batch = faces_embeds_batch.to(device)
            print(f"faces_embeds_batch.shape: {faces_embeds_batch.shape} device: {faces_embeds_batch.device}")
            #print(f"image_proj_model.device:" {})
            # pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample

            # if len(res_frame_list) == 0:
            #     print("using first frame")
            #     last_frame = crop_frames[0]
            # else:
            #     print("using last frame")
            #     last_frame = res_frame_list[-1]
            
            # faces = face_embedding_extractor.get(last_frame)
            # if len(faces) != 0:
            #     print("using faces")
            #     faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                
            # print(f"faceid_embeds: {faceid_embeds}")
            # faceid_embeds_batch_list = [faceid_embeds] * len(latent_batch)
            # print(f"faceid_embeds_batch_list: {faceid_embeds_batch_list}")

            
            # faceid_embeds_batch_new = torch.cat(faceid_embeds_batch_list, dim=0)
            # faceid_embeds_batch_new = faceid_embeds_batch_new.to(device=unet.device, dtype=unet.model.dtype)

            # pred_latents = ip_adapter(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch, image_embeds=faces_embeds_batch)
            pred_latents = ip_adapter(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch, image_embeds=faces_embeds_batch)
            
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
            # if i % 2 == 0:
            #     attn_maps = get_net_attn_map((256, 256), batch_size=8)
            #     #print(attn_maps[0].shape)
            #     attn_hot = attnmaps2images(attn_maps)
            #     # print(f"attn_hot.shape:{attn_hot[0].shape}")
            #     # concatenated_images = concat_images(recon, attn_hot)
            #     fig, axes = plt.subplots(1, len(attn_hot), figsize=(12, 4))
            #     for axe, image in zip(axes, attn_hot):
            #         axe.imshow(image, cmap='gray')
            #         axe.axis('off')

            #     # Save the final concatenated image
            #     plt.savefig(f"{atten_map_img_save_path}/concatenated_attention_map_{i}.png")
            
            # filtered_est_matched_images = [img for img in best_matched_images_batch if img is not None]
            # total_width = len(filtered_est_matched_images) * RESIZED_IMG
            # if len(filtered_est_matched_images) > 0:
            #     combined_image = Image.new('RGB', (total_width, RESIZED_IMG))
            #     x_offset = 0
            #     for img in filtered_est_matched_images:
            #         combined_image.paste(img, (x_offset, 0))
            #         x_offset += RESIZED_IMG
            #     combined_image.save(os.path.join(result_img_save_path, f"best_matched_images_batch_{i}.png"))

            # to compare
            # filtered_crop_images = [img for img in crop_images_batch if img is not None]
            # total_width = len(filtered_crop_images) * RESIZED_IMG
            # if len(filtered_crop_images) > 0:
            #     combined_image = Image.new('RGB', (total_width, RESIZED_IMG))
            #     x_offset = 0
            #     for img in filtered_crop_images:
            #         combined_image.paste(img, (x_offset, 0))
            #         x_offset += RESIZED_IMG
            #     combined_image.save(os.path.join(result_img_save_path, f"original_images_batch_{i}.png"))

                
        ############################################## pad to full image ##############################################
        print("pad talking image to original video")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
#             bbox = coord_list_cycle[i%(len(coord_list_cycle))]
#             ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
#             x1, y1, x2, y2 = bbox
#             try:
#                 res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
#             except:
# #                 print(bbox)
#                 continue
            
#             combine_frame = get_image(ori_frame,res_frame,bbox)
            # cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", res_frame)

        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p -crf 18 temp.mp4"
        print(cmd_img2video)
        os.system(cmd_img2video)
        
        cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i temp.mp4 {output_vid_name}"
        print(cmd_combine_audio)
        os.system(cmd_combine_audio)
        
        os.remove("temp.mp4")
        # shutil.rmtree(result_img_save_path)
        print(f"result is save to {output_vid_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/inference/test.yaml")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--result_dir", default='./results', help="path to output")

    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_vid_name", type=str, default=None)
    parser.add_argument("--use_saved_coord",
                        default=True,
                        action="store_true",
                        help='use saved coordinate to save time')
    parser.add_argument("--use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default="train_codes/output/adapter0104/checkpoint-20000")
    # "train_codes/output/adapter1130/checkpoint-40000"
    # parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None)
    parser.add_argument("--preprocessing",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    parser.add_argument("--crop_path", type=str, default=None)
    parser.add_argument("--face_embs_path", type=str)
    parser.add_argument("--guide_scale", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
