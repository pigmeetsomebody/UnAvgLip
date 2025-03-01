import insightface
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from musetalk.utils.utils import get_file_type
from musetalk.utils.preprocessing import read_imgs
import glob
import os
import torch
import torch
import torch.nn.functional as F
# 加载 ArcFace 模型
def load_model():
    # 加载 InsightFace 的 ArcFace 模型
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0, det_size=(256, 128))  # ctx_id=0 表示使用 CPU，det_size 设置为 256x256
    return model

# 加载模型
model = load_model()

# 提取视频中的人脸嵌入向量
def extract_face_embedding_from_video(video_path, model):
    video = cv2.VideoCapture(video_path)
    embeddings = []
    
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 获取视频帧的人脸特征
        faces = model.get(frame)
        if faces:
            # 假设每个视频帧只包含一个人脸，取第一个人脸的嵌入向量
            face = faces[0]
            embedding = face.embedding  # 人脸嵌入向量
            embeddings.append(embedding)
    
    video.release()
    return embeddings




import librosa
import numpy as np
import cv2

# 获取音频的时间戳
def get_audio_time_stamps(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    # 提取梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y, sr=sr)
    # 计算梅尔频谱的时间戳
    times = librosa.times_like(mel_spec)
    return times

# 获取视频的帧时间戳
def get_video_time_stamps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    time_stamps = np.arange(0, total_frames) / fps
    video.release()
    return time_stamps

# 基于音频和视频的时间戳对齐视频帧
def align_video_frames_based_on_audio(video_path, audio_path):
    audio_time_stamps = get_audio_time_stamps(audio_path)
    video_time_stamps = get_video_time_stamps(video_path)

    video = cv2.VideoCapture(video_path)
    aligned_frames = []
    
    for audio_time in audio_time_stamps:
        closest_frame_idx = np.argmin(np.abs(video_time_stamps - audio_time))
        video.set(cv2.CAP_PROP_POS_FRAMES, closest_frame_idx)
        ret, frame = video.read()
        if ret:
            aligned_frames.append(frame)
    
    video.release()
    return aligned_frames

import cv2
import numpy as np

def align_frames(real_video_path, generated_video_path):
    # 读取真实视频和生成视频
    real_video = cv2.VideoCapture(real_video_path)
    generated_video = cv2.VideoCapture(generated_video_path)

    # 获取视频帧率
    fps_real = real_video.get(cv2.CAP_PROP_FPS)
    fps_generated = generated_video.get(cv2.CAP_PROP_FPS)

    # 计算帧率差异
    ratio = fps_real / fps_generated

    # 获取视频的总帧数
    total_frames_real = int(real_video.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_generated = int(generated_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化视频帧列表
    aligned_real_frames = []
    aligned_generated_frames = []

    # 生成帧索引对应关系
    real_frame_indices = np.arange(0, total_frames_real, step=1)
    generated_frame_indices = np.arange(0, total_frames_generated, step=1/ratio)
    
    generated_frame_indices = generated_frame_indices.astype(int)

    # 提取并对齐帧
    for idx_real, idx_generated in zip(real_frame_indices, generated_frame_indices):
        real_video.set(cv2.CAP_PROP_POS_FRAMES, idx_real)
        ret_real, frame_real = real_video.read()
        if not ret_real:
            break

        generated_video.set(cv2.CAP_PROP_POS_FRAMES, idx_generated)
        ret_generated, frame_generated = generated_video.read()
        if not ret_generated:
            break

        aligned_real_frames.append(frame_real)
        aligned_generated_frames.append(frame_generated)

    real_video.release()
    generated_video.release()

    return aligned_real_frames, aligned_generated_frames


# aligned_real_frames, aligned_generated_frames = align_frames(real_video_path, generated_video_path)

import cv2

def get_generated_frames_from_video(generated_video_path):
    # Open the video file
    generated_video = cv2.VideoCapture(generated_video_path)
    
    # Get total number of frames in the video
    total_frames = int(generated_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize an empty list to store the frames
    generated_frames = []
    
    for idx in range(0, total_frames):
        # Set the video frame position to the idx-th frame
        generated_video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        
        # Read the frame
        ret_generated, frame_generated = generated_video.read()
        
        if ret_generated:
            generated_frames.append(frame_generated)
        else:
            print(f"Warning: Failed to read frame {idx}")
            break

    # Release the video capture object
    generated_video.release()
    
    return generated_frames


#real_frames, generated_frames = align_frames(real_video_path, generated_video_path)

# 示例用法
# 获取视频帧率
# 提取原始视频和生成视频中的人脸嵌入
real_video_path = 'data/video/talk.mp4'
generated_video_path = 'results/zhy_zhy_adapter_False_1.0.mp4'
generated_video_path2 = 'results/zhy_zhy.mp4'
real_video = cv2.VideoCapture(real_video_path)
generated_video = cv2.VideoCapture(generated_video_path)
fps_real = real_video.get(cv2.CAP_PROP_FPS)
fps_generated = generated_video.get(cv2.CAP_PROP_FPS)
# 计算帧率差距
ratio = fps_real / fps_generated

real_video.release()
generated_video.release()

# 初始化视频帧列表
aligned_real_frames = []
aligned_generated_frames = []


generated_frames = get_generated_frames_from_video(generated_video_path)
generated_frames2 = get_generated_frames_from_video(generated_video_path2)
real_video_frame_path = "results/zhy_zhy/zhy_zhy"
real_crop_img_list = glob.glob(os.path.join(real_video_frame_path, '*.[jpJP][pnPN]*[gG]'))
real_crop_img_list = sorted(real_crop_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
real_frames = read_imgs(real_crop_img_list)
ratio = len(real_frames) / len(generated_frames)
print(f"fps_real: {fps_real}, fps_generated:{fps_generated}, ratio: {ratio}")
# 生成帧索引对应关系
real_frame_indices = np.arange(0, len(real_frames), step=1)
generated_frame_indices = np.arange(0, len(generated_frames), step=1/ratio)
print(f"Aligned Real Video Frames: {len(real_frame_indices)}")
print(f"Aligned Generated Video Frames: {len(generated_frame_indices)}")

real_face_embeds, generated_face_embeds, generated_face_embeds2, similaritys, similaritys2  = [], [], [],[],[]
for idx_real, idx_generated in zip(real_frame_indices, generated_frame_indices):
    idx_generated = int(idx_generated)
    real_frame = real_frames[idx_real]
    height = real_frame.shape[1]
    y1, y2 = height // 2, height
    real_frame = real_frame[y1:y2, :]
    generated_frame = generated_frames[idx_generated]
    generated_frame = generated_frame[y1:y2, :]
    generated_frame2 = generated_frames2[idx_generated]
    generated_frame2 = generated_frame2[y1:y2, :]

    real_faces = model.get(real_frame)
    if len(real_faces) == 0:
        print("no face in real")
        continue
    generated_faces = model.get(generated_frame)
    if len(generated_faces) == 0:
        print("no face in generated")
        continue
    generated_faces2 = model.get(generated_frame2)
    if len(generated_faces2) == 0:
        print("no face in generated")
        continue
    real_face_embs = torch.from_numpy(real_faces[0].normed_embedding).unsqueeze(0)
    generated_faces_embs = torch.from_numpy(generated_faces[0].normed_embedding).unsqueeze(0)
    generated_faces_embs2 = torch.from_numpy(generated_faces2[0].normed_embedding).unsqueeze(0)
    similarity = F.cosine_similarity(real_face_embs, generated_faces_embs)
    similarity2 = F.cosine_similarity(real_face_embs, generated_faces_embs2)
    print(f"similarity:{similarity.item()}, similarity2:{similarity2.item()}")
    real_face_embeds.append(real_face_embs)
    generated_faces.append(generated_faces_embs)
    similaritys.append(similarity.item())
    similaritys2.append(similarity2.item())
avg_similarity1 = np.mean(similaritys)
avg_similarity2 = np.mean(similaritys2)
improved = (avg_similarity1 - avg_similarity2) / avg_similarity2
print(f"avg_similarity1: {np.mean(similaritys)}, avg_similarity2: {np.mean(similaritys2)}, improved:{improved}")


# # 示例用法
# audio_path = 'results/talk.wav'
# generated_video_path = 'results/silence_talk_adapter_False_1.0.mp4'
# actual_video_path = "data/video/talk.mp4"
# generated_aligned_video_frames = align_video_frames_based_on_audio(generated_video_path, audio_path)
# actual_aligned_video_frames = align_video_frames_based_on_audio(actual_video_path, audio_path)
# print(f"Aligned generated_aligned_video_frames Frames: {len(generated_aligned_video_frames)}, actual_aligned_video_frames: {len(actual_aligned_video_frames)}")

# real_face_embeds_audio, generated_face_embeds_audio = [], []
# for real_video_frame, generated_video_frame in zip(actual_aligned_video_frames, generated_aligned_video_frames):
#     real_faces = model.get(real_video_frame)
#     if len(real_faces) == 0:
#         continue
#     generated_faces = model.get(generated_video_frame)
#     if len(generated_faces) == 0:
#         continue
#     real_face_embeds_audio.append(real_faces[0])
#     generated_face_embeds_audio.append(generated_faces[0])



def compute_cosine_similarity(real_face_embeds, generated_face_embeds):
    similarities = []
    
    for real_embed, generated_embed in zip(real_face_embeds, generated_face_embeds):
        # Ensure the embeddings are tensors
        real_embed = torch.tensor(real_embed).unsqueeze(0)  # Add batch dimension
        generated_embed = torch.tensor(generated_embed).unsqueeze(0)  # Add batch dimension
        
        # Compute cosine similarity between real and generated face embeddings
        similarity = F.cosine_similarity(real_embed, generated_embed)
        print(similarity.item())
        similarities.append(similarity.item())  # Convert tensor to Python float
    
    return similarities


# 计算两个嵌入向量之间的平均余弦相似度
def compute_identity_consistency(real_embeddings, generated_embeddings):
    similarities = []
    
    # 假设两个视频的帧数相同，按帧对应计算相似度
    for real_emb, generated_emb in zip(real_embeddings, generated_embeddings):
        similarity = 1 - cosine(real_emb, generated_emb)  # 余弦相似度 (1 - cosine 距离)
        similarities.append(similarity)
    
    # 计算所有帧的平均相似度
    avg_similarity = np.mean(similarities)
    return avg_similarity


# identity_consistency_1 = compute_cosine_similarity(real_face_embeds, generated_face_embeds)
#identity_consistency_2 = compute_identity_consistency(real_face_embeds_audio, generated_face_embeds_audio)

# print(f"identity_consistency_2: {identity_consistency_2}")
# 计算身份一致性
# identity_consistency = compute_identity_consistency(real_face_embeddings, generated_face_embeddings)
# print(f"Identity Consistency: {identity_consistency}")
