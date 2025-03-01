import os
import cv2
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision import transforms
import copy
from skimage.metrics import structural_similarity as ssim

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

    
from musetalk.whisper.audio2feature import Audio2Feature
from musetalk.models.vae import VAE
from musetalk.models.unet import UNet,PositionalEncoding
import insightface
from  insightface.app import FaceAnalysis

def load_all_model():
    audio_processor = Audio2Feature(model_path="./models/whisper/tiny.pt")
    vae = VAE(model_path = "./models/sd-vae-ft-mse/")
    unet = UNet(unet_config="./models/musetalk/musetalk.json",
                model_path ="./models/musetalk/pytorch_model.bin")
    pe = PositionalEncoding(d_model=384)
    return audio_processor,vae,unet,pe

def get_file_type(video_path):
    _, ext = os.path.splitext(video_path)

    if ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        return 'image'
    elif ext.lower() in ['.avi', '.mp4', '.mov', '.flv', '.mkv']:
        return 'video'
    else:
        return 'unsupported'

def get_video_fps(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

codebook = None

def codebook_gen(kmeans, knn, whisper_chunks, vae_encode_latents, delay_frame=0):
    concatenated_features = []  # Initialize concatenated features list
    for i, w in enumerate(whisper_chunks):
        # idx = (i+delay_frame)%len(vae_encode_latents)
        # latent = vae_encode_latents[idx]
        # print(f"whisper_batch.shape: {w.shape}, latent_batch.shape: {latent.shape}")
        # flatten the image featuer
        # flattened_image_latent = latent.view(1, -1).cpu()
        
        audio_flat = np.mean(w, axis=0, keepdims=True)
        audio_flat_tensor = torch.tensor(audio_flat, dtype=torch.float32).cpu()
        # concatenated_feature = torch.cat((audio_flat_tensor, flattened_image_latent), dim=1).cpu()
        # print(f"flattened_image_latent.shape: {flattened_image_latent.shape}, audio_flat_tensor.shape: {audio_flat_tensor.shape}, concatenated_feature.shape: {concatenated_feature.shape}")
        concatenated_features.append(audio_flat_tensor)
    concatenated_features = np.vstack(concatenated_features)
    # clustering
    n_clusters = min(3600, len(concatenated_features))
    # kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42)
    kmeans.fit(concatenated_features)
    centroids = kmeans.cluster_centers_
    # knn = KNeighborsClassifier(n_neighbors=5)  # Nearest centroid
    # print(f"kmeans.cluster_centers_: {centroids}")
    knn.fit(centroids, np.arange(n_clusters))


def knn_codebook_gen(kmeans, knn, whisper_chunks, vae_encode_latents, delay_frame=0):
    concatenated_features = []  # Initialize concatenated features list
    for i, w in enumerate(whisper_chunks):
        # idx = (i+delay_frame)%len(vae_encode_latents)
        # latent = vae_encode_latents[idx]
        # print(f"whisper_batch.shape: {w.shape}, latent_batch.shape: {latent.shape}")
        # flatten the image featuer
        # flattened_image_latent = latent.view(1, -1).cpu()
        
        audio_flat = np.mean(w, axis=0, keepdims=True)
        audio_flat_tensor = torch.tensor(audio_flat, dtype=torch.float32).cpu()
        # concatenated_feature = torch.cat((audio_flat_tensor, flattened_image_latent), dim=1).cpu()
        # print(f"flattened_image_latent.shape: {flattened_image_latent.shape}, audio_flat_tensor.shape: {audio_flat_tensor.shape}, concatenated_feature.shape: {concatenated_feature.shape}")
        concatenated_features.append(audio_flat_tensor)
    concatenated_features = np.vstack(concatenated_features)
    # clustering
    # n_clusters = min(3600, len(concatenated_features))
    # kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=100, random_state=42)
    # kmeans.fit(concatenated_features)
    # centroids = kmeans.cluster_centers_
    # knn = KNeighborsClassifier(n_neighbors=5)  # Nearest centroid
    # print(f"kmeans.cluster_centers_: {centroids}")
    labels = np.arange(len(concatenated_features))
    knn.fit(concatenated_features, labels)
def calculate_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

RESIZED_IMG = 256


def calculate_laplacian_variance(image):
    image_gray = np.array(image.convert('L'))  # Convert to grayscale
    laplacian_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()  # Calculate Laplacian variance
    return laplacian_var

def compute_ssim(imageA, imageB):
    # Convert images from RGB to Grayscale if they are in (256, 256, 3) shape
    grayA = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    
    # Compute SSIM (Structural Similarity Index)
    ssim_value, _ = ssim(grayA, grayB, full=True)
    return ssim_value

def search_audios_pair_images(kmeans, knn, whisper_chunks, crop_image_list, save_dir):
    results = []
    face_embs_results = []
    embeds_map = {}
    face_embedding_extractor = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_embedding_extractor.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG))

    pre_candidate = -1
    for i, w in enumerate(whisper_chunks):
        new_audio_flat = np.mean(w, axis=0, keepdims=True)  # Shape (1, 384)
        new_audio_flat_tensor = torch.tensor(new_audio_flat, dtype=torch.float32).cpu()
        # Create a dummy image feature (optional)
        # latent = torch.rand(1, 8, 32, 32)
        # flattened_image_latent = latent.view(1, -1)
        # Concatenate new audio and dummy image features
        # new_concatenated_feature = torch.cat((new_audio_flat_tensor, flattened_image_latent), dim=1)  # Shape (1, 8576)
        
        #if len(predicted_clusters) > 
        distances, indices = knn.kneighbors(new_audio_flat_tensor, n_neighbors=5)
        if len(indices[0]) > 0:
            print(f"knn predicted kneighbors: {indices[0]}, crop_image_list: {len(crop_image_list)}")
            retrieved_images = [crop_image_list[idx-1] for idx in indices[0]]
            max_ssim = -1 
            now_candidate = indices[0][0] - 1
            for i, img in enumerate(retrieved_images):
                if pre_candidate == -1:
                    now_candidate = indices[0][0] - 1
                    break
                else:
                    current_ssim = compute_ssim(img, crop_image_list[pre_candidate])
                    print(f"SSIM between current and previous candidate: {current_ssim}")
                    if current_ssim < max_ssim:
                        max_ssim = current_ssim
                        now_candidate = indices[0][i] - 1
            result_image = copy.deepcopy(crop_image_list[now_candidate])
            if now_candidate not in embeds_map:
                faces = face_embedding_extractor.get(result_image)
                print(f"detect faces: {len(faces)}")
                if len(faces) <= 0:
                    faceid_embed = torch.zeros(1, 512)
                else:
                    faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                embeds_map[now_candidate] = faceid_embed
            print(f"select now candidate: {now_candidate}, pre_candidate: {pre_candidate}")
            faceid_embed = embeds_map[now_candidate]
            pre_candidate = now_candidate
            result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            results.append(result_image)
            face_embs_results.append(faceid_embed)
            continue


        # indices will give you the indices of the 5 nearest neighbors in the original dataset
        print(f"Indices of 5 nearest neighbors: {indices}")
        print(f"Distances to the 5 nearest neighbors: {distances}")

        # Get predicted cluster using KNN
        predicted_clusters = knn.predict(new_audio_flat_tensor)
        print(f"Predicted clusters for audio {i}: {predicted_clusters}")
        predicted_cluster = predicted_clusters[0]

        
        if  predicted_cluster >= 0 and predicted_cluster <= len(crop_image_list):
            print(f"Predicted cluster for audio {i}: {predicted_cluster}")
            result_image = crop_image_list[predicted_cluster - 1]
            if predicted_cluster not in embeds_map:
                faces = face_embedding_extractor.get(result_image)
                print(f"detect faces: {len(faces)}")
                if len(faces) <= 0:
                    faceid_embed = torch.zeros(1, 512)
                else:
                    faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                embeds_map[predicted_cluster] = faceid_embed
            result_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            faceid_embed = embeds_map[predicted_cluster]
            results.append(result_image)
            face_embs_results.append(faceid_embed)
            continue


        
        # Retrieve the cluster labels
        cluster_labels = kmeans.labels_
        cluster_indices = np.where(cluster_labels == predicted_cluster)[0]
        print(f"Cluster indices: {cluster_indices}, crop_image_list: {len(crop_image_list)}")

        if len(cluster_indices) > 0:
            # Get the image features that belong to the predicted cluster
            # valid_indices = [idx for idx in cluster_indices if 0 <= idx < len(crop_image_list)]
            retrieved_images = [crop_image_list[idx-1] for idx in cluster_indices]
            if len(retrieved_images) <= 0:
                result_image = None
                results.append(result_image)
                faceid_embed = torch.zeros(1, 512)
                face_embs_results.append(faceid_embed)
                continue

            # retrieved_images = [crop_image_list[idx] for idx in cluster_indices]
            
            # print(f"len of retrieved_images: {len(retrieved_images)}")
            # Initialize variables to track the clearest image
            result_image = retrieved_images[0]
            result_cluster_indice = cluster_indices[0]
            max_variance = -1
            save_images = Image.new('RGB', (RESIZED_IMG * (len(cluster_indices) + 1), RESIZED_IMG))
            j = 0
            result_index = j
            for image in retrieved_images:
                
                # Calculate the Laplacian variance for the image
                # variance = calculate_laplacian_variance(image)
                # print(f"Laplacian variance for image {i}: {variance}")
                #image_transposed = np.transpose(image, (2, 0, 1))
                # print(f"image.shape: {image.shape}")
                if isinstance(image, torch.Tensor):
                    # Convert PyTorch tensor to PIL Image
                    transform = transforms.ToPILImage()
                    image = transform(image)
                elif isinstance(image, np.ndarray):
                    
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                # print(f"i: {i}, image.shape: {image.shape}, image_transposed: {image_transposed.shape}")
                # Paste the image into the combined visualization
                save_images.paste(image, (j * RESIZED_IMG, 0))
                # Calculate the Laplacian variance for the image
                variance = calculate_laplacian_variance(image)
                # Update the clearest image if this one has a higher variance
                if variance > max_variance:
                    max_variance = variance
                    result_image = image
                    result_cluster_indice = cluster_indices[j]
                    result_index = j
                j += 1

            
            # Paste the clearest image at the end of the concatenated image for clarity
            save_images.paste(result_image, (j * RESIZED_IMG, 0))
            
            # Ensure save directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the image with a unique filename for the batch
            val_img_dir = f"{save_dir}/{i}_matched_images.png"
            save_images.save(val_img_dir)
            if result_cluster_indice in embeds_map:
                faceid_embed = embeds_map[result_cluster_indice]
            else:
                faces = face_embedding_extractor.get(retrieved_images[result_index])
                print(f"detect faces: {len(faces)}")
                if len(faces) <= 0:
                    faceid_embed = torch.zeros(1, 512)
                else:
                    faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                embeds_map[result_cluster_indice] = faceid_embed
        else:
            result_image = None
            faceid_embed = torch.zeros(1, 512)
        
        results.append(result_image)
        face_embs_results.append(faceid_embed)
        print(f"best matched face embeds: {faceid_embed.shape}")
    return results, face_embs_results



def datagen(whisper_chunks,
            vae_encode_latents,
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch = [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        whisper_batch.append(w)
        latent_batch.append(latent)

        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            yield whisper_batch, latent_batch
            whisper_batch, latent_batch = [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        yield whisper_batch, latent_batch
        

# def datagen_with_matched_faces(whisper_chunks,
#             vae_encode_latents,
#             matched_faces=[],
#             batch_size=8,
#             delay_frame=0):
    

def datagen_with_faceid(whisper_chunks,
            vae_encode_latents,
            faces_embeds=[],
            best_matched_image=[],
            crop_images=[],
            batch_size=8,
            delay_frame=0):
    whisper_batch, latent_batch, faces_embeds_batch, best_matched_image_batch, crop_images_batch = [], [], [], [], []
    for i, w in enumerate(whisper_chunks):
        idx = (i+delay_frame)%len(vae_encode_latents)
        latent = vae_encode_latents[idx]
        faces_embed = faces_embeds[idx]
        crop_image = copy.deepcopy(crop_images[idx])
        crop_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
        faces_embeds_batch.append(faces_embed)            
        whisper_batch.append(w)
        latent_batch.append(latent)
        crop_images_batch.append(crop_image)
        # best_matched_image_batch.append(best_matched_image[idx])
        if len(latent_batch) >= batch_size:
            whisper_batch = np.stack(whisper_batch)
            latent_batch = torch.cat(latent_batch, dim=0)
            faces_embeds_batch = torch.cat(faces_embeds_batch, dim=0)
            yield whisper_batch, latent_batch, faces_embeds_batch, best_matched_image_batch, crop_images_batch
            whisper_batch, latent_batch, faces_embeds_batch, best_matched_image_batch, crop_images_batch = [], [], [], [], []

    # the last batch may smaller than batch size
    if len(latent_batch) > 0:
        whisper_batch = np.stack(whisper_batch)
        latent_batch = torch.cat(latent_batch, dim=0)
        faces_embeds_batch = torch.cat(faces_embeds_batch, dim=0)     
        yield whisper_batch, latent_batch, faces_embeds_batch, best_matched_image_batch, crop_images_batch