import sys
from face_detection import FaceAlignment,LandmarksType
from os import listdir, path
import subprocess
import numpy as np
import cv2
import pickle
import os
import json
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
import torch
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import glob
import cv2
from insightface.app import FaceAnalysis
from scipy.spatial.distance import pdist, squareform
import numpy as np

def select_most_different_landmarks(landmarks, num_to_select=5):
    """
    Select the `num_to_select` most different landmarks from the given list of landmarks.
    
    Parameters:
        landmarks (list or np.ndarray): A list or array of landmarks (each is a set of points).
        num_to_select (int): Number of diverse landmarks to select.
    
    Returns:
        selected_indices (list): Indices of the selected landmarks.
    """
    # Calculate pairwise distances between all landmarks
    distances = squareform(pdist(landmarks, metric="euclidean"))  # Pairwise Euclidean distances
    
    # Start with the first landmark
    selected_indices = [0]  # Start with an arbitrary landmark (e.g., first one)
    
    while len(selected_indices) < num_to_select:
        max_min_distance = -1
        next_index = -1
        
        # Find the landmark that is farthest from the already selected ones
        for i in range(len(landmarks)):
            if i in selected_indices:
                continue  # Skip already selected landmarks
            
            # Compute the minimum distance to the selected landmarks
            min_distance = min(distances[i][j] for j in selected_indices)
            
            # Keep track of the landmark with the maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                next_index = i
        
        # Add the next most diverse landmark
        selected_indices.append(next_index)
    
    return selected_indices


def load_face_net_model(modelpath):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.io.gfile.GFile(modelpath, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# initialize the mmpose model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_file = './musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
checkpoint_file = './models/dwpose/dw-ll_ucoco_384.pth'
face_net_model_path = './models/face_net/20170511-185253.pb'
model = init_model(config_file, checkpoint_file, device=device)
face_net = load_face_net_model(face_net_model_path)
# initialize the face detection model
device = "cuda" if torch.cuda.is_available() else "cpu"
fa = FaceAlignment(LandmarksType._2D, flip_input=False,device=device)

# maker if the bbox is not sufficient 
coord_placeholder = (0.0,0.0,0.0,0.0)


def get_face_embedding(imgs):
    h, w, c = imgs[0].shape
    images = np.zeros((len(imgs), h, w, 3))
    
    for i, crop_frame in enumerate(imgs):
        # resized = cv2.resize(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB),(int(256),int(256))) 
        images[i,:,:,:] = crop_frame
        #cv2.imwrite(path.join(save_dir, '{}.png'.format(i)),full_frames[i][0][y1:y2, x1:x2])
    print(f"the shape of images: {images.shape}")
    with face_net.as_default():
        with tf.Session() as sess:
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            embeddings = sess.run(embeddings, feed_dict=feed_dict)
    print(f"the shape of embeddings: {embeddings.shape}")
    return embeddings

def resize_landmark(landmark, w, h, new_w, new_h):
    w_ratio = new_w / w
    h_ratio = new_h / h
    landmark_norm = landmark / [w, h]
    landmark_resized = landmark_norm * [new_w, new_h]
    return landmark_resized

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

def get_bbox_range(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
    text_range=f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}"
    return text_range

def find_minimal_morpha_lip(frames):
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    landmarks = []
    i = 0
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        range_v_bound = abs((face_land_mark[66]- face_land_mark[62])[1])
        range_h_bound = abs((face_land_mark[54]- face_land_mark[48])[1])
        if range_v_bound < 2 and range_h_bound < 2:
            return i     
        i += 1
    return 0
            




def get_landmark_and_bbox(img_list,upperbondrange =0):
    frames = read_imgs(img_list)
    batch_size_fa = 1
    batches = [frames[i:i + batch_size_fa] for i in range(0, len(frames), batch_size_fa)]
    coords_list = []
    landmarks = []
    if upperbondrange != 0:
        print('get key_landmark and face bounding boxes with the bbox_shift:',upperbondrange)
    else:
        print('get key_landmark and face bounding boxes with the default value')
    average_range_minus = []
    average_range_plus = []
    face_land_marks = []
    for fb in tqdm(batches):
        results = inference_topdown(model, np.asarray(fb)[0])
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark= keypoints[0][23:91]
        face_land_mark = face_land_mark.astype(np.int32)
        face_land_marks.append(face_land_mark)
        # get bounding boxes by face detetion
        bbox = fa.get_detections_for_batch(np.asarray(fb))
        
        # adjust the bounding box refer to landmark
        # Add the bounding box to a tuple and append it to the coordinates list
        for j, f in enumerate(bbox):
            if f is None: # no face in the image
                coords_list += [coord_placeholder]
                continue
            
            half_face_coord =  face_land_mark[29]#np.mean([face_land_mark[28], face_land_mark[29]], axis=0)
            range_minus = (face_land_mark[30]- face_land_mark[29])[1]
            range_plus = (face_land_mark[29]- face_land_mark[28])[1]
            average_range_minus.append(range_minus)
            average_range_plus.append(range_plus)
            if upperbondrange != 0:
                half_face_coord[1] = upperbondrange+half_face_coord[1] #手动调整  + 向下（偏29）  - 向上（偏28）
            half_face_dist = np.max(face_land_mark[:,1]) - half_face_coord[1]
            upper_bond = half_face_coord[1]-half_face_dist
            
            f_landmark = (np.min(face_land_mark[:, 0]),int(upper_bond),np.max(face_land_mark[:, 0]),np.max(face_land_mark[:,1]))
            x1, y1, x2, y2 = f_landmark
            
            if y2-y1<=0 or x2-x1<=0 or x1<0: # if the landmark bbox is not suitable, reuse the bbox
                coords_list += [f]
                w,h = f[2]-f[0], f[3]-f[1]
                print("error bbox:",f)
            else:
                coords_list += [f_landmark]
    
    print("********************************************bbox_shift parameter adjustment**********************************************************")
    print(f"Total frame:「{len(frames)}」 Manually adjust range : [ -{int(sum(average_range_minus) / len(average_range_minus))}~{int(sum(average_range_plus) / len(average_range_plus))} ] , the current value: {upperbondrange}")
    print("*************************************************************************************************************************************")
    return coords_list,frames, face_land_marks

def extract_audio_from_video(video_path, output_audio_path):
    # FFmpeg命令
    command = [
        'ffmpeg',             # 调用ffmpeg命令
        '-i', video_path,     # 输入视频文件路径
        '-vn',                # 禁用视频流
        '-acodec', 'pcm_s16le',  # 设置音频编解码器为pcm_s16le（WAV格式）
        '-ar', '44100',       # 设置音频采样率为44.1kHz
        '-ac', '2',           # 设置音频通道数为2（立体声）
        output_audio_path     # 输出音频文件路径
    ]
    
    try:
        # 执行命令
        subprocess.run(command, check=True)
        print(f"Audio extracted successfully: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while extracting audio: {e}")

if __name__ == "__main__":
    img_list = glob.glob(os.path.join('./results/RD_Radio18_000_val_part0/RD_Radio18_000_val_part0/', '*.[jpJP][pnPN]*[gG]'))
    img_list = sorted(img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    # coords_list,full_frames, lms = get_landmark_and_bbox(img_list)
    full_frames = read_imgs(img_list)
    crop_coord_path = './RD_Radio18_000_val_part0.pkl'
    lms_coord_path = './RD_Radio18_000_val_part0_landmarks.pkl'
    # with open(crop_coord_path, 'wb') as f:
    #     pickle.dump(coords_list, f)
    # # 保存 lms 到文件
    # with open(lms_coord_path, "wb") as f:
    #     pickle.dump(lms, f)
    # print("Landmarks saved to landmarks.pkl")

    with open(crop_coord_path, "rb") as f:
        coords_list = pickle.load(f)
        print("coords_list loaded successfully")
    # 从文件加载 lms
    with open(lms_coord_path, "rb") as f:
        lms = pickle.load(f)
        print("Landmarks loaded successfully")
    crop_frames = []
    num_faces = len(coords_list)

    min_diff = float('inf')
    min_idx = -1
    target_rect = (0, 0, 0, 0)
    for idx, landmarks in enumerate(lms):
        range_plus = (landmarks[66]- landmarks[62])[1]
        m_x = landmarks[66]- landmarks[62]

        lip_coords = landmarks[48:60]
        # 计算边界框
        min_x = int(np.min(lip_coords[:, 0]))
        max_x = int(np.max(lip_coords[:, 0]))
        min_y = int(np.min(lip_coords[:, 1]))
        max_y = int(np.max(lip_coords[:, 1]))
        if range_plus < min_diff:
            min_diff = range_plus
            min_idx = idx
            target_rect = (min_x, min_y, max_x, max_y)
            print(f"range_plus: {range_plus}")
            print(f"target_rect: {target_rect}")

    print(f"min_diff of mouth: {min_idx}")
    
    for bbox, frame in zip(coords_list,full_frames):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        #print('Cropped shape', crop_frame.shape)
        resized = cv2.resize(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB),(int(160),int(160))) 
        crop_frames.append(resized)

    height, width, channels = crop_frames[0].shape
    images = np.zeros((len(crop_frames), height, width, 3))


    target_img = full_frames[min_idx]
    x1, y1, x2, y2 = target_rect
    buffer = 50  # Add padding
    print(f"target_rect: {target_rect}")
    print(f"target_img shape: {target_img.shape}")  # Should be (height, width, channels)
    x1 = max(0, min(x1 - buffer, target_img.shape[1]))
    x2 = max(0, min(x2 + buffer, target_img.shape[1]))
    y1 = max(0, min(y1 - buffer, target_img.shape[0]))
    y2 = max(0, min(y2 + buffer, target_img.shape[0]))

    lip_region = target_img[y1:y2, x1:x2]  # Cropping the region


    height = target_img.shape[1]
    # Define the lip region as the lower half
    y1, y2 = height // 2, height
    cropped_lip_region = target_img[y1:y2, :]

    # Save cropped region
    cv2.imwrite("lip_region.jpg", lip_region)
    cv2.imwrite("target_image.jpg", cropped_lip_region)
    # 初始化 InsightFace
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(256, 256))
    faces = app.get(cropped_lip_region)

    if len(faces) == 0:
        print("No faces detected in target_img.")
        lip_feature = None
    else:
        lip_feature = faces[0].normed_embedding
    print("Lip feature extracted successfully.")
    # lip_feature = app.get(cropped_lip_region)[0].normed_embedding
    print(f"min_diff of mouth: {min_idx}")

    if lip_feature is not None:
        print(f"lip_feature: {lip_feature.shape}")
        face_embeddings_path = "RD_Radio18_000_val_part0_face_embeddings.npy"
        np.save("RD_Radio18_000_val_part0_face_embeddings.npy", lip_feature)
        loaded_embeddings = np.load(face_embeddings_path)
        print("Loaded embeddings shape:", loaded_embeddings.shape)

    target_index = find_minimal_morpha_lip(full_frames)
    print(f"find_minimal_morpha_lip: {target_index}")
    # indices = select_most_different_landmarks(lms)
    # print(f"select_most_different_landmarks indices:{indices}")
    

