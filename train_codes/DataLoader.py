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
import PIL.Image
import math

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        if not os.path.exists(img_path):
            continue
        frame = cv2.imread(img_path)
        height, width = frame.shape[:2]
        if height % 2 != 0 or width % 2 != 0:
            new_height = (height + 1) // 2 * 2
            new_width = (width + 1) // 2 * 2
            frame = cv2.resize(frame, (new_width, new_height))
        frames.append(frame)
    return frames

syncnet_T = 5
RESIZED_IMG = 256
mel_sample_rate = 16000
n_cond = 1
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

face_model = "antelopev2"
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


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

        
class Dataset(object):
    def __init__(self, 
                 data_root, 
                 json_path, 
                 use_audio_length_left=1,
                 use_audio_length_right=1,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05,
                 whisper_model_type = "tiny",
                 n_face_id_cond = 1,
                 ):
        # self.all_videos, self.all_imgNum = get_image_list(data_root, split)
        self.audio_feature = [use_audio_length_left,use_audio_length_right]
        self.all_img_names = []
        # self.split = split
        self.img_names_path = '../new_train_data'
        self.whisper_model_type = whisper_model_type
        self.use_audio_length_left = use_audio_length_left
        self.use_audio_length_right = use_audio_length_right
        # self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app = FaceAnalysis(name=face_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG))
        self.total_img_nums = 0
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.face_id_embeds_paths = []
        self.data_root = data_root
        self.n_face_id_cond = n_face_id_cond
        self.conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ]
        )

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
        valid_videos = []
        for vidname in tqdm(self.all_videos, desc="Preparing dataset"):
            vid_full_path = vidname
            vidname = vidname.split('/')[-1].split('.')[0]
            video_path = os.path.join(f"{self.img_names_path}/images", vidname)
            face_kps_dir = os.path.join(data_root, "face_kps/")
            if self.n_face_id_cond == 1:
                face_emb_dir = os.path.join(data_root, "face_id_full/")
            else:
                face_emb_dir = os.path.join(data_root, f"face_id_full_{self.n_face_id_cond}/")
            json_path_names = f"{self.img_names_path}/{vidname}.json"
            # os.makedirs(face3dmm_save_dir, exist_ok=True)
            os.makedirs(face_emb_dir, exist_ok=True)
            os.makedirs(face_kps_dir, exist_ok=True)
            # print(f"save face3dmm processed images in: {face3dmm_save_dir}")
            # if not os.path.exists(json_path_names):
            print(f"json_path_names: {json_path_names}, video_path:{video_path}")
            if not os.path.exists(json_path_names):
                print(f"json_path_names: {json_path_names}, video_path:{video_path}")
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
            
            images = read_imgs(img_names)
            if len(img_names) == 0 or len(images) == 0:
                continue
            
            valid_videos.append(vid_full_path)
            print(f"add valid_video: {vid_full_path}")
            self.total_img_nums += len(img_names)
            self.all_img_names.append(img_names)
            

            face_embs_path = os.path.join(face_emb_dir, vidname + '.pt')
            face_kps_path = os.path.join(face_kps_dir, vidname + '.png')
            if not os.path.exists(face_embs_path) or not os.path.exists(face_kps_path):
                print(f"extract face embeds: {face_embs_path}")
                face_embs_path = self.selected_random_face_embedings(images, n_face_id_cond, face_emb_dir, vidname)
                face_img = images[0]
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                if face_model == "antelopev2":
                    face_info = self.app.get(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))[-1]
                    face_emb = face_info['embedding']
                    face_emb = torch.from_numpy(face_emb)
                    face_emb = face_emb.unsqueeze(0)
                    kps = face_info['kps']
                else:
                    faces = self.app.get(face_img)
                    face_emb = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                    kps = faces[0].kps
                # torch.save(face_emb, face_embs_path)
                
                print(f"face kps: {len(kps)}")
                face_kps = draw_kps(face_pil, kps)
                print(f"face_kps.size:{face_kps.size} face_embs: {face_emb.shape}")
                face_kps.save(face_kps_path)
                print(f"save face embeds in: {face_embs_path}, face_kps_path: {face_kps_path}")
            else:
                print(f"using existing face embeds: {face_embs_path}")
            self.face_id_embeds_paths.append(face_embs_path)
        print(f"valid videos: {len(self.all_videos)}")
        self.all_videos = valid_videos
            
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
            face_info = self.app.get(random_frame)[-1]
            face_emb = face_info['embedding']
            if face_emb is None:
                print("no face detect!")
                continue
            else:
                print(f"face_embed.shape: {face_emb.shape}")
                # face_kps = draw_kps(face_img, face_info['kps'])
                # face_kps.save(face_kps)
                image.paste(face_img, (RESIZED_IMG*i, 0))
                image_path = os.path.join(save_dir, f"{vidname}.png")
                face_embs.append(torch.from_numpy(face_emb).unsqueeze(0))
                i += 1
        image.save(image_path)
        face_embeds_ts = torch.cat(face_embs, dim=1)
        face_emb_path = os.path.join(save_dir, f"{vidname}.pt")
        torch.save(face_embeds_ts, face_emb_path)
        return face_emb

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
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x
    
    def get_segmented_mels(self, spec, start_frame, fps):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2, fps)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels
    
    def crop_audio_window(self, spec, start_frame, fps):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame) + 1  # 0-indexing ---> 1-indexing

        num = syncnet_T * syncnet_mel_step_size
        start_idx = int(80. * (start_frame_num / float(fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx: end_idx, :]
    
    def get_audio_feature_window(self, start_id, sub_folder_name):
        audio_feature_all = []
        audio_features = []
        is_index_out_of_range = False
        for frame_id in range(start_id, start_id + syncnet_T):
            if os.path.isdir(os.path.join(self.whisper_path, sub_folder_name)):
                for feat_idx in range(frame_id-self.use_audio_length_left,frame_id+self.use_audio_length_right+1):
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
            audio_feature_all = []
            audio_feature = audio_feature.reshape(1, -1, self.whisper_feature_H) #1， -1， 384
            if audio_feature.shape != (1, self.whisper_feature_concateW, self.whisper_feature_H):  #1 50 384
                print(f"shape error!! {sub_folder_name} {frame_id}, audio_feature.shape: {audio_feature.shape}")
                continue
            audio_features.append(audio_feature)
        if len(audio_features) != syncnet_T:
            return None
        audio_features_np = np.concatenate(audio_features, axis=0)
        return audio_features_np




    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            #随机选择某个video里
            vidname = self.all_videos[idx].split('/')[-1]
            video_imgs = self.all_img_names[idx]
            imgs_dir = self.all_videos[idx]

            if len(video_imgs) == 0:
                print("video_imgs = 0:",vidname)
                continue
            img_name = random.choice(video_imgs)
            img_idx = int(basename(img_name).split(".")[0])
            random_element = random.randint(0,len(video_imgs)-1)
            while abs(random_element - img_idx) <= 5:
                random_element = random.randint(0,len(video_imgs)-1)
            ref_image = os.path.join(imgs_dir, f"{str(random_element)}.png")
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
            image = gt = target_window.copy()
            target_window[:, :, target_window.shape[2]//2:] = 0.                   # upper half face, mask掉下半部分        V1：输入       
            ref_image = self.prepare_window(ref_window)
            ref_image = torch.FloatTensor(ref_image)
            mask = torch.zeros_like(ref_image)
            
            mask[:, :, :ref_image.shape[2]//2,:] = 1
            image = torch.FloatTensor(image)
            # ref_image = torch.FloatTensor(ref_image)
            # mask, masked_image = prepare_mask_and_masked_image(ref_image,mask)

            masks_prepared = []
            masked_images = []
            for i in range(ref_image.shape[1]):
                img = ref_image[:, i, :, :]      # Shape: (C, H, W)
                m = mask[:, i, :, :]             # Shape: (C, H, W)
    
                # Apply the masking function
                m_prepared, masked = prepare_mask_and_masked_image(img, m)
    
                # Append results to the lists
                masks_prepared.append(m_prepared)
                masked_images.append(masked)
            
            masked_images = torch.stack(masked_images, dim=1)  # Shape: (C, N, H, W)
            masks_prepared = torch.stack(masks_prepared, dim=1)  # Shape: (C, N, H, W)
            # 人脸特征
            face_ids_dir = os.path.join(self.data_root, 'face_id_full')
            face_kps_dir = os.path.join(self.data_root, 'face_kps')
            faceid_embeds_path = os.path.join(face_ids_dir, vidname + '.pt')
            face_kps_path = os.path.join(face_kps_dir, vidname + '.png')
            if os.path.exists(faceid_embeds_path):
                faceid_embeds = torch.load(faceid_embeds_path)
            else:
                faceid_embeds = torch.zeros(1, 5, 512)
            
            if os.path.exists(face_kps_path):
                kps_image = Image.open(face_kps_path)
            else:
                kps_image = None
            
            # selected_face_emb = faceid_embeds[0, 0:n_cond, :]  # shape [1, 512]
            if faceid_embeds.dim() == 2:
                selected_face_emb = faceid_embeds
            else:
                selected_face_emb = faceid_embeds[0, 0:n_cond, :]  # shape [1, 512]
            selected_face_embs = selected_face_emb.repeat(syncnet_T, 1, 1)
            kps_tensor = self.conditioning_image_transforms(kps_image)
            kps_tensor = kps_tensor.unsqueeze(0).repeat(syncnet_T, 1, 1, 1)
            # selected_face_emb = selected_face_emb.unsqueeze(0)
            # drop
            drop_image_embed = 0
            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_image_embed = 1
            if drop_image_embed:
                selected_face_embs = torch.zeros_like(selected_face_embs)
            
            #音频特征
            window_index = self.get_frame_id(img_name)
            sub_folder_name = vidname.split('/')[-1]
            
            ## 根据window_index加载相邻的音频
            # audio_feature_all = []
            # is_index_out_of_range = False
            # if os.path.isdir(os.path.join(self.whisper_path, sub_folder_name)):
            #     for feat_idx in range(window_index-self.use_audio_length_left,window_index+self.use_audio_length_right+1):
            #         # 判定是否越界
            #         audio_feat_path = os.path.join(self.whisper_path, sub_folder_name, str(feat_idx) + ".npy")
            #         if not os.path.exists(audio_feat_path):
            #             is_index_out_of_range = True
            #             break

            #         try:
            #             audio_feature_all.append(np.load(audio_feat_path))
            #         except Exception as e:
            #             print(f"发生未知错误：{e}")
            #             print(f"npy load error {audio_feat_path}")
            #     if is_index_out_of_range:
            #         continue
            #     audio_feature = np.concatenate(audio_feature_all, axis=0)
            # else:
            #     continue

            # audio_feature = audio_feature.reshape(1, -1, self.whisper_feature_H) #1， -1， 384
            # if audio_feature.shape != (1,self.whisper_feature_concateW, self.whisper_feature_H):  #1 50 384
            #     print(f"shape error!! {vidname} {window_index}, audio_feature.shape: {audio_feature.shape}")
            #     continue
            # audio_feature = torch.squeeze(torch.FloatTensor(audio_feature))
            audio_features = self.get_audio_feature_window(window_index, sub_folder_name)
            if audio_features is None: continue
            audio_features = torch.squeeze(torch.FloatTensor(audio_features))
            audio_features = audio_features.squeeze(0)

            # # mel频谱
            try:
                wav_path = f"{self.whisper_path}/{vidname}.wav"
                wav = load_wav(wav_path, mel_sample_rate)
                orig_mel = melspectrogram(wav).T
                
            except Exception as e:
                print(e)
                continue
            duration = len(orig_mel) / mel_sample_rate
            fps = len(video_imgs) / duration

            mel = self.crop_audio_window(orig_mel.copy(), img_name, fps)
            if (mel.shape[0] != syncnet_mel_step_size):
                print(f"mel shape {mel.shape[0]} didn't match syncnet_mel_step_size:{syncnet_mel_step_size}")
                continue

            indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name, fps)
            if indiv_mels is None: continue
            indiv_mels = torch.FloatTensor(indiv_mels)
            # mel = self.crop_audio_window(orig_mel.copy(), img_name, fps)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)
            # mel = []
            ref_image = ref_image.squeeze(0)  # 删除第一个维度
            ref_image = ref_image.permute(1, 0, 2, 3)  # 交换维度0和维度1，得到[5, 3, 256, 256]
            image = image.squeeze(0)  # 删除第一个维度
            image = image.permute(1, 0, 2, 3)  # 交换维度0和维度1，得到[5, 3, 256, 256]
            masked_images = masked_images.squeeze(0)  # 删除第一个维度
            masked_images = masked_images.permute(1, 0, 2, 3)  # 交换维度0和维度1，得到[5, 3, 256, 256]
            masks_prepared = masks_prepared.squeeze(0)  # 删除第一个维度
            masks_prepared = masks_prepared.permute(1, 0, 2, 3)  # 交换维度0和维度1，得到[5, 3, 256, 256]

            return ref_image, image, masked_images, masks_prepared, audio_features, selected_face_embs, mel, kps_tensor
         
    
    
if __name__ == "__main__":
    data_root = '../new_train_data'
    val_data = Dataset(data_root, 
                          '../test.json', 
                          use_audio_length_left = 2,
                          use_audio_length_right = 2,
                          whisper_model_type = "tiny",
                          n_face_id_cond=5
                          )
    val_data_loader = data_utils.DataLoader(
        val_data, batch_size=2, shuffle=True,
        num_workers=1)

    for i, data in enumerate(val_data_loader):
        ref_image, image, masked_image, mask, audio_features, face_emb, mel, kps = data
        print(f"ref_image.shape: {ref_image.shape}, image.shape: {image.shape}, mask.shape: {mask.shape}, masked_image.shape: {masked_image.shape}, audio_features.shape: {audio_features.shape}, face_emb:{face_emb.shape}, mel:{mel.shape} face_kps: {kps.shape}")
        if kps is None:
            print(f"kps image is None")
        else:
            print(f"kps image size: {kps.shape}")

 