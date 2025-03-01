import os, random, cv2, argparse
import torch
from torch.utils import data as data_utils
from os.path import dirname, join, basename, isfile
import numpy as np
from glob import glob
# train_codes
from utils.utils import prepare_mask_and_masked_image
import torchvision.utils as vutils
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm
import ast
import json
import re
import heapq
import insightface
from insightface.app import FaceAnalysis
from utils.audio import load_wav, melspectrogram, hp
import sys
# from face3dmm.utils.safetensor_helper import load_x_from_safetensor 
# from face3dmm.face3d.extract_kp_videos_safe import KeypointExtractor
# from face3dmm.utils.init_path import init_path
# from face3dmm.face3d.util.generate_facerender_batch import get_facerender_data_simple
# from face3dmm.facerender.animate import AnimateFromCoeff
# from face3dmm.preprocessing import preprocessing_images
from PIL import Image

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        height, width = frame.shape[:2]
        if height % 2 != 0 or width % 2 != 0:
            new_height = (height + 1) // 2 * 2
            new_width = (width + 1) // 2 * 2
            frame = cv2.resize(frame, (new_width, new_height))
        frames.append(frame)
    return frames

syncnet_T = 1
RESIZED_IMG = 256
syncnet_mel_step_size = 16
connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),(7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),(13,14),(14,15),(15,16),  # 下颌线
                       (17, 18), (18, 19), (19, 20), (20, 21), #左眉毛
                       (22, 23), (23, 24), (24, 25), (25, 26), #右眉毛
                       (27, 28),(28,29),(29,30),# 鼻梁
                       (31,32),(32,33),(33,34),(34,35), #鼻子
                       (36,37),(37,38),(38, 39), (39, 40), (40, 41), (41, 36), # 左眼
                       (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42), # 右眼
                       (48, 49),(49, 50), (50, 51),(51, 52),(52, 53), (53, 54), # 上嘴唇 外延
                       (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # 下嘴唇 外延
                       (60, 61), (61, 62), (62, 63), (63, 64), (64, 65),  (65, 66), (66, 67), (67, 60) #嘴唇内圈
              ]  


def get_image_list(data_root, split):
    filelist = []
    imgNumList = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                filename = line.split()[0]
                imgNum = int(line.split()[1])
                filelist.append(os.path.join(data_root, filename))
            imgNumList.append(imgNum)
    return filelist, imgNumList


drop_image_embed = 0
rand_num = random.random()
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


        

        
class Dataset(object):
    def __init__(self, 
                 data_root, 
                 json_path, 
                 use_audio_length_left=1,
                 use_audio_length_right=1,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 whisper_model_type = "tiny",
                 n_face_id_cond = 5,
                 ):
        # self.all_videos, self.all_imgNum = get_image_list(data_root, split)
        self.audio_feature = [use_audio_length_left,use_audio_length_right]
        self.all_img_names = []
        # self.split = split
        self.img_names_path = '../new_train_data'
        self.whisper_model_type = whisper_model_type
        self.use_audio_length_left = use_audio_length_left
        self.use_audio_length_right = use_audio_length_right
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG))
        self.total_img_nums = 0
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.face_id_embeds_paths = []
        self.data_root = data_root
        self.still_imgs_dirs = []

        if self.whisper_model_type =="tiny":
            self.whisper_path = '../new_train_data/audios'
            self.whisper_feature_W = 5
            self.whisper_feature_H = 384
        elif self.whisper_model_type =="largeV2":
            self.whisper_path = '...'
            self.whisper_feature_W = 33
            self.whisper_feature_H = 1280
        self.whisper_feature_concateW = self.whisper_feature_W*2*(self.use_audio_length_left+self.use_audio_length_right+1) #5*2*（2+2+1）= 50
        print(f"load train data set from : {json_path}")
        with open(json_path, 'r') as file:
            self.all_videos = json.load(file)

        for vidname in tqdm(self.all_videos, desc="Preparing dataset"):
            vidname = vidname.split('/')[-1].split('.')[0]
            video_path = os.path.join(f"{self.img_names_path}/images", vidname)
            face_emb_dir = os.path.join(data_root, "face_id_full/")
            still_imgs_dir = os.path.join(data_root, "still_images/")
            still_imgs_video_dir = os.path.join(os.path.join(still_imgs_dir, vidname), "images")
            self.still_imgs_dirs.append(still_imgs_video_dir)
            json_path_names = f"{self.img_names_path}/{vidname}.json"
            # face3dmm_save_dir = f"{self.img_names_path}/still_images/{vidname.split('/')[-1].split('.')[0]}/"
            # os.makedirs(face3dmm_save_dir, exist_ok=True)
            os.makedirs(face_emb_dir, exist_ok=True)
            # print(f"save face3dmm processed images in: {face3dmm_save_dir}")
            # if not os.path.exists(json_path_names):
            if not os.path.exists(json_path_names):
                img_names = glob(join(video_path, '*.png'))
                img_names.sort(key=lambda x:int(x.split("/")[-1].split('.')[0]))
                # images = read_imgs(img_names)
                # pil_frames = [Image.open(image_path) for image_path in img_names]
                # 处理face_embeddings_tensor
                # face_embs_path = self.selected_random_face_embedings(images, n_face_id_cond, face_emb_dir, vidname)

                # 查找唇动嘴小的
                # minimal_mouth_diff_index = find_minimal_morpha_lip(images)

                # To animate video to silent mode
                # best_ref_img = pil_frames[minimal_mouth_diff_index]
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # crop_frames_dir = preprocessing_images(best_ref_img, pil_frames, face3dmm_save_dir, vidname, device)
                # crop_img_names = glob(join(crop_frames_dir, '*.png'))
                # crop_img_names.sort(key=lambda x:int(x.split("/")[-1].split('.')[0]))

                with open(json_path_names, "w") as f:
                    json.dump(img_names,f)
            else:
                with open(json_path_names, "r") as f:
                    img_names = json.load(f)
            
            self.total_img_nums += len(img_names)
            self.all_img_names.append(img_names)
            images = read_imgs(img_names)
            if len(img_names) == 0:
                continue
            face_embs_path = os.path.join(face_emb_dir, vidname + '.pt')
            if not os.path.exists(face_embs_path):
                print(f"extract face embeds: {face_embs_path}")
                face_embs_path = self.selected_random_face_embedings(images, n_face_id_cond, face_emb_dir, vidname)
            else:
                print(f"using existing face embeds: {face_embs_path}")
            self.face_id_embeds_paths.append(face_embs_path)
            
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])
    
    def selected_random_face_embedings(self, frames, target_num, save_dir, vidname):
        i = 0
        face_embs = []
        # for debug
        image = Image.new('RGB', (RESIZED_IMG*target_num, RESIZED_IMG))
        while i < target_num:
            random_frame = random.choice(frames)
            height = random_frame.shape[1]
            # Define the lip region as the lower half
            # y1, y2 = height // 2, height
            # cropped_lip_region = random_frame[y1:y2, :]
            face_img = Image.fromarray(cv2.cvtColor(random_frame, cv2.COLOR_BGR2RGB))
            faces = self.app.get(random_frame)
            if len(faces) <= 0:
                print("no face detect!")
                continue
            else:
                image.paste(face_img, (RESIZED_IMG*i, 0))
                image_path = os.path.join(save_dir, f"{vidname}.png")
                image.save(image_path)
                face_embs.append(torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).unsqueeze(0))
                i += 1
        face_embeds_ts = torch.cat(face_embs, dim=1)
        face_emb_path = os.path.join(save_dir, f"{vidname}.pt")
        torch.save(face_embeds_ts, face_emb_path)
        return face_emb_path

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.png'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames
    
    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (RESIZED_IMG, RESIZED_IMG))
            except Exception as e:
                print("read_window has error fname not exist:",fname)
                return None

            window.append(img)

        return window
    
    def prepare_window(self, window):
        #  1 x H x W x 3
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x
    
    def crop_audio_window(self, spec, start_frame, fps):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(fps)))

        print(f"crop_audio_window spec.shape: {spec.shape}, start_idx: {start_idx}, hp.fps: {fps}")
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            print("__getitem__")
            idx = random.randint(0, len(self.all_videos) - 1)
            #随机选择某个video里
            vidname = self.all_videos[idx].split('/')[-1]
            video_imgs = self.all_img_names[idx]
            still_imgs_dir = self.still_imgs_dirs[idx]

            if len(video_imgs) == 0:
#                 print("video_imgs = 0:",vidname)
                continue
            img_name = random.choice(video_imgs)
            img_idx = int(basename(img_name).split(".")[0])
            random_element = random.randint(0,len(video_imgs)-1)
            while abs(random_element - img_idx) <= 5:
                random_element = random.randint(0,len(video_imgs)-1)
            ref_image = os.path.join(still_imgs_dir, f"{str(random_element)}.png")
            target_window_fnames = self.get_window(img_name)
            ref_window_fnames = self.get_window(ref_image)

            if target_window_fnames is None or ref_window_fnames is None:
                print("No such img",img_name, ref_image)
                continue
            try:
                #构建目标img数据
                target_window = self.read_window(target_window_fnames)
                if target_window is None :
                    print("No such target window,",target_window_fnames)
                    continue
                #构建参考img数据
                ref_window = self.read_window(ref_window_fnames)
    
                if ref_window is None:
                    print("No such target ref window,",ref_window)
                    continue
            except Exception as e:
                print(f"发生未知错误：{e}")
                continue
          
            #构建target输入
            target_window = self.prepare_window(target_window)
            image = gt = target_window.copy().squeeze()
            target_window[:, :, target_window.shape[2]//2:] = 0.                   # upper half face, mask掉下半部分        V1：输入       
            ref_image = self.prepare_window(ref_window).squeeze()   
            mask = torch.zeros((ref_image.shape[1], ref_image.shape[2]))
            mask[:ref_image.shape[2]//2,:] = 1
            image = torch.FloatTensor(image)
            ref_image = torch.FloatTensor(ref_image)
            mask, masked_image = prepare_mask_and_masked_image(ref_image,mask)
           
            # 人脸特征
            face_ids_dir = os.path.join(self.data_root, 'face_id_full')
            faceid_embeds_path = os.path.join(face_ids_dir, vidname + '.pt')
            if os.path.exists(faceid_embeds_path):
                faceid_embeds = torch.load(faceid_embeds_path)
            else:
                faceid_embeds = torch.zeros(1, 5, 512)
            idx = torch.randint(5, (1,)) 
            print(f"faceid_embeds.shape: {faceid_embeds.shape}")
            selected_face_emb = faceid_embeds[0, idx, :]  # shape [1, 512]
            # selected_face_emb = selected_face_emb.unsqueeze(0)
            print(f"selected_face_emb.shape: {selected_face_emb.shape}")
            # drop
            drop_image_embed = 0
            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_image_embed = 1
            if drop_image_embed:
                selected_face_emb = torch.zeros_like(selected_face_emb)
            
            #音频特征
            window_index = self.get_frame_id(img_name)
            sub_folder_name = vidname.split('/')[-1]
            
            ## 根据window_index加载相邻的音频
            audio_feature_all = []
            is_index_out_of_range = False
            if os.path.isdir(os.path.join(self.whisper_path, sub_folder_name)):
                for feat_idx in range(window_index-self.use_audio_length_left,window_index+self.use_audio_length_right+1):
                    # 判定是否越界
                    audio_feat_path = os.path.join(self.whisper_path, sub_folder_name, str(feat_idx) + ".npy")
                    if not os.path.exists(audio_feat_path):
                        is_index_out_of_range = True
                        break

                    try:
                        audio_feature_all.append(np.load(audio_feat_path))
                    except Exception as e:
                        print(f"发生未知错误：{e}")
                        print(f"npy load error {audio_feat_path}")
                if is_index_out_of_range:
                    continue
                audio_feature = np.concatenate(audio_feature_all, axis=0)
            else:
                continue

            audio_feature = audio_feature.reshape(1, -1, self.whisper_feature_H) #1， -1， 384
            if audio_feature.shape != (1,self.whisper_feature_concateW, self.whisper_feature_H):  #1 50 384
                print(f"shape error!! {vidname} {window_index}, audio_feature.shape: {audio_feature.shape}")
                continue
            audio_feature = torch.squeeze(torch.FloatTensor(audio_feature))

            # # mel频谱
            # try:
            #     wav_path = f"{self.whisper_path}/{vidname}.wav"
            #     wav = load_wav(wav_path, hp.sample_rate)
            #     orig_mel = melspectrogram(wav).T

            # except Exception as e:
            #     print(e)
            #     continue
            # duration = len(orig_mel) / hp.sample_rate
            # fps = len(video_imgs) / duration
            # mel = self.crop_audio_window(orig_mel.copy(), img_name, fps)

            # if (mel.shape[0] != syncnet_mel_step_size):
            #     print(f"load melspectrogram of {wav_path} failed. mel:shape:{mel.shape}")
            #     continue
            mel = []
            return ref_image, image, masked_image, mask, audio_feature, selected_face_emb, mel
         
    
    
if __name__ == "__main__":
    data_root = '../new_train_data'
    val_data = Dataset(data_root, 
                          '../train.json', 
                          use_audio_length_left = 2,
                          use_audio_length_right = 2,
                          whisper_model_type = "tiny"
                          )
    val_data_loader = data_utils.DataLoader(
        val_data, batch_size=256, shuffle=True,
        num_workers=1)

    for i, data in enumerate(val_data_loader):
        ref_image, image, masked_image, mask, audio_feature, face_emb, mel = data
        print(f"ref_image.shape: {ref_image.shape}")

 