import torch
import numpy as np
from pytorch_msssim import ssim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import cv2
# from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import euclidean
import librosa
import argparse
import os
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from PIL import Image
from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import read_imgs, get_landmark_and_bbox, coord_placeholder
import glob
import insightface
import torch.nn.functional as F
import pickle
import json


class FaceGenerationMetrics:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Initialize face detection and recognition models
        # self.mtcnn = MTCNN(device=device)
        self.facenet = insightface.app.FaceAnalysis()
        self.facenet.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(256, 256))

        # self.facenet.prepare(ctx_id=0, det_size=(256, 128))
        # self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        
    def compute_visual_quality(self, generated_imgs, reference_imgs):
        """
        Compute visual quality metrics including SSIM and PSNR
        Args:
            generated_img: Generated image tensor (B, C, H, W)
            reference_img: Reference image tensor (B, C, H, W)
        Returns:
            Dictionary containing SSIM and PSNR scores
        """
        ssim_list, psnr_list = [], []
        for generated_img, reference_img in zip(generated_imgs, reference_imgs):
            # Ensure inputs are torch tensors
            # if not isinstance(generated_img, torch.Tensor):
            #     generated_img = torch.from_numpy(generated_img).to(self.device)
            # if not isinstance(reference_img, torch.Tensor):
            #     reference_img = torch.from_numpy(reference_img).to(self.device)
            # print(f"generated_img.shape:{generated_img.shape}, reference_img:{reference_img.shape}")
            if not isinstance(generated_img, torch.Tensor):
                generated_img = torch.from_numpy(generated_img).to(self.device).float()
            if not isinstance(reference_img, torch.Tensor):
                reference_img = torch.from_numpy(reference_img).to(self.device).float()

            # Normalize if the images are in the range [0, 255]
            generated_img = generated_img / 255.0
            reference_img = reference_img / 255.0
            # print(f"generated_img.dim():{generated_img.dim(), reference_img.dim()}")

            # Reorder dimensions from (256, 256, 3) to (3, 256, 256) and add batch dimension (1, 3, 256, 256)
            generated_img = generated_img.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 256, 256]
            reference_img = reference_img.permute(2, 0, 1).unsqueeze(0)  # Shape: [1, 3, 256, 256]
            
            # Compute SSIM
            ssim_score = ssim(generated_img, reference_img, data_range=1.0)
        
            # Compute PSNR
            mse = F.mse_loss(generated_img, reference_img)
            psnr = 10 * torch.log10(1.0 / mse)
            ssim_list.append(ssim_score.item())
            psnr_list.append(psnr.item())
            # print(f"ssim_score: {ssim_score.item()}, psnr: {psnr.item()}")
        
        avg_ssim_score = np.mean(ssim_list)
        avg_psnr_score = np.mean(psnr_list)
        # print(f"avg_ssim_score: {avg_ssim_score}, avg_psnr_score:{avg_psnr_score}")
        
        return {
            'avg_ssim_score': avg_ssim_score,
            'avg_psnr_score': avg_psnr_score
        }

    def get_sample_indices(
        self,
        total_frames: int,
        strategy: str = 'uniform',
        num_samples: int = 10,
        sample_rate: int = 10,
        keyframes = []
    ) -> np.ndarray:
        """
        Get indices for sampling frames while maintaining correspondence.
        
        Args:
            total_frames (int): Total number of frames
            strategy (str): Sampling strategy:
                - 'uniform': Uniformly spaced samples
                - 'random': Random sampling
                - 'keyframe': Sample at specified keyframes
                - 'fixed_interval': Sample every nth frame
            num_samples (int): Number of frames to sample (for uniform/random)
            sample_rate (int): Sample every nth frame (for fixed_interval)
            keyframes (List[int]): List of specific frame indices to sample (for keyframe)
            
        Returns:
            np.ndarray: Array of indices to sample
        """
        if strategy == 'uniform':
            if num_samples > total_frames:
                num_samples = total_frames
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
        elif strategy == 'random':
            if num_samples > total_frames:
                num_samples = total_frames
            indices = np.sort(np.random.choice(total_frames, num_samples, replace=False))
            
        elif strategy == 'keyframe':
            if not keyframes:
                raise ValueError("Keyframes list must be provided for keyframe strategy")
            indices = np.array([i for i in keyframes if i < total_frames])
            
        elif strategy == 'fixed_interval':
            if not sample_rate:
                raise ValueError("Sample rate must be provided for fixed_interval strategy")
            indices = np.arange(0, total_frames, sample_rate)
            
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
            
        return indices

    
    def compute_identity_consistency(self, gt_frames, generated_frames, sampling_params = None):
        """
        Compute identity consistency between ground truth and generated frames.
        
        Args:
            gt_frames (numpy.ndarray): Ground truth frames of shape (N, H, W, C)
            generated_frames (numpy.ndarray): Generated frames of shape (N, H, W, C)
            
        Returns:
            dict: Dictionary containing average cosine similarity and L2 distance.
                Returns None if no valid face pairs are found.
        """

        total_frames = len(gt_frames)
        if len(gt_frames) != len(generated_frames):
            total_frames = min(len(gt_frames), len(generated_frames))
            # raise ValueError(f"Length mismatch: gt_frames {len(gt_frames)} != generated_frames {len(generated_frames)}")
        if sampling_params:
            # print(f"sampling_params: {sampling_params} sampled indices: {len(indices)}")
            indices = self.get_sample_indices(total_frames, **sampling_params)
            # print(f"sampling_params: {sampling_params} sampled indices: {len(indices)}")
            frames_to_process = [(i, gt_frames[i], generated_frames[i]) for i in indices]
            # gt_frames = gt_frames[indices]
            # generated_frames = generated_frames[indices]
        else:
            frames_to_process = [(i, gt_frames[i], generated_frames[i]) for i in range(total_frames)]
            # original_indices = indices
        # else:
        #     # original_indices = np.arange(len(gt_frames))
        # print(f"sampling_params: {sampling_params} sampled indices: {len(indices)}, total_frames:{total_frames}")
        cosine_similarities = []
        l2_distances = []
        
        for idx, gt_frame, generated_frame in frames_to_process:
            gen_embedding = self.extract_face_embedding(generated_frame)
            ref_embedding = self.extract_face_embedding(gt_frame)
            if gen_embedding is None or ref_embedding is None:
                continue
                
            cosine_sim = F.cosine_similarity(ref_embedding, gen_embedding).item()
            l2_dist = euclidean(
                gen_embedding.cpu().numpy().flatten(),
                ref_embedding.cpu().numpy().flatten()
            )
            
            cosine_similarities.append(cosine_sim)
            l2_distances.append(l2_dist)
        
        if not cosine_similarities:
            # print("Warning: No valid face pairs found in the frames")
            return None
            
        return {
            'avg_cosine_similarity': np.mean(cosine_similarities),
            'avg_l2_distance': np.mean(l2_distances),
            'std_cosine_similarity': np.std(cosine_similarities),
            'std_l2_distance': np.std(l2_distances),
            'num_valid_pairs': len(cosine_similarities)
        }
    
    def extract_face_embedding(self, img):
        """
        Extract face embedding from the lower half of an image.
        
        Args:
            img (numpy.ndarray): Input image of shape (H, W, C)
            
        Returns:
            torch.Tensor: Face embedding tensor of shape (1, embedding_dim)
            None: If no face is detected
        """
        try:
            # height = img.shape[0]
            # y1, y2 = height // 2, height
            # half_img = img[y1:y2, :]
            
            faces = self.facenet.get(img)
            
            if not faces:
                return None
                
            face_embs = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
            return face_embs.to(self.device)
            
        except Exception as e:
            # print(f"Error extracting face embedding: {str(e)}")
            return None
    
    def compute_lip_sync_quality(self, video_frames, audio, fps=30):
        """
        Compute lip sync quality metrics
        Args:
            video_frames: List of video frames (T, H, W, C)
            audio: Audio waveform
            fps: Video frames per second
        Returns:
            Dictionary containing sync offset and confidence scores
        """
        # Extract lip landmarks for each frame
        lip_movements = []
        for frame in video_frames:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Extract lip region and compute movement
            face = self.mtcnn.detect(frame)
            if face is not None:
                bbox = face[0][0]
                lip_region = gray[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                movement = np.mean(np.abs(np.diff(lip_region.flatten())))
                lip_movements.append(movement)
        
        # Extract audio features
        audio_features = librosa.feature.melspectrogram(y=audio, sr=16000)
        audio_energy = np.mean(audio_features, axis=0)
        
        # Compute correlation between lip movements and audio energy
        correlation = np.correlate(
            lip_movements, 
            audio_energy[:len(lip_movements)], 
            mode='full'
        )
        
        max_correlation_idx = np.argmax(correlation)
        sync_offset = (max_correlation_idx - len(lip_movements)) / fps
        sync_confidence = np.max(correlation) / np.mean(correlation)
        
        return {
            'sync_offset': sync_offset,
            'sync_confidence': sync_confidence
        }

    def evaluate_model(self, generated_samples, reference_samples, audio_samples=None):
        """
        Comprehensive evaluation of the model
        Args:
            generated_samples: List of generated images/videos
            reference_samples: List of reference images/videos
            audio_samples: Optional list of audio samples for lip sync evaluation
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {
            'visual_quality': [],
            'identity_consistency': [],
            'lip_sync': [] if audio_samples else None
        }
        # print("begin evaluating visual_quality....")
        results['visual_quality'] = self.compute_visual_quality(generated_samples, reference_samples)
        # print("begin evaluating identity_consistency....")
        # Sample every 10th frame
        sampling_params = {
            'strategy': 'fixed_interval',
            'sample_rate': 100
        }
        results['identity_consistency'] = self.compute_identity_consistency(generated_samples, reference_samples)
        # for gen, ref in zip(generated_samples, reference_samples):
        #     print(f"gen.shape: {gen.shape}, ref.shape: {ref.shape}")
        #     results['visual_quality'].append(
                
        #     )
        #     results['identity_consistency'].append(
        #         self.compute_identity_consistency(gen, ref)
        #     )
            
        # if audio_samples:
        #     for gen, audio in zip(generated_samples, audio_samples):
        #         results['lip_sync'].append(
        #             self.compute_lip_sync_quality(gen, audio)
        #         )
                
        return results


def align_images(gt_img_list, generated_img_list):
    real_frames = read_imgs(gt_img_list)
    generated_frames = read_imgs(generated_img_list)
    ratio = len(gt_img_list) / len(generated_img_list)
    # print(f"gt_img_list: {len(gt_img_list)}, generated_img_list: {len(generated_img_list)}, ratio:{ratio}")
    aligned_real_frames, aligned_generated_frames = [], []
    real_frame_indices = np.arange(0, len(real_frames), step=1)
    generated_frame_indices = np.arange(0, len(generated_frames), step=1/ratio)
    for idx_real, idx_generated in zip(real_frame_indices, generated_frame_indices):
        idx_generated = int(idx_generated)
        # print(f"generated_img: {generated_img_list[idx_generated]}, generated_shape: ")
        # if generated_frames[idx_generated].shape[0] != 256 or generated_frames[idx_generated].shape[1] != 256:
        generated_frames[idx_generated] = cv2.resize(generated_frames[idx_generated],(256,256),interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(generated_img_list[idx_generated], generated_frames[idx_generated])
        aligned_real_frames.append(real_frames[idx_real])
        aligned_generated_frames.append(generated_frames[idx_generated])
    return aligned_real_frames, aligned_generated_frames



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssims = []
psnrs = []
cosine_similaritys = []
@torch.no_grad()
def main(args):
    # print(f"args.evaluate_config: {args.evaluate_config}")
    evaluate_config = OmegaConf.load(args.evaluate_config)
    print(f"evaluate_config: {evaluate_config}")
    metrics = FaceGenerationMetrics()
    res_str = ""
    for task_id in evaluate_config:
        gt_video_path = evaluate_config[task_id]["gt_video_path"]
        generated_video_path = evaluate_config[task_id]["generated_video_path"]
        result_save_path = evaluate_config[task_id]["result_save_path"]
        gt_cropped = False
        generated_cropped = False
        # generated_cropped = evaluate_config[task_id]["generated_cropped"]
        # gt_cropped = evaluate_config[task_id]["gt_cropped"]

        # input_basename = task_id
        
        # print(task_id)
       
        # save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(result_save_path, exist_ok = True)
        # handle gt_video_path
        if get_file_type(gt_video_path)=="video":
            save_gt_dir = os.path.join(result_save_path, "gt")
            os.makedirs(save_gt_dir,exist_ok = True)
            cmd = f"ffmpeg -v fatal -i {gt_video_path} -start_number 0 {save_gt_dir}/%08d.png"
            os.system(cmd)
            # coord_list, frame_list, face_land_marks = get_landmark_and_bbox(input_img_list, bbox_shift)
            gt_img_list = sorted(glob.glob(os.path.join(save_gt_dir, '*.[jpJP][pnPN]*[gG]')))
        elif get_file_type(gt_video_path)=="image":
            gt_img_list = [gt_video_path, ]
        elif os.path.isdir(gt_video_path):
            gt_img_list = glob.glob(os.path.join(gt_video_path, '*.[jpJP][pnPN]*[gG]'))
            gt_img_list = sorted(gt_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            # raise ValueError(f"{gt_video_path} should be a video file, an image file or a directory of images")
            continue
            print(f"{gt_video_path} should be a video file, an image file or a directory of images")
        
        if gt_cropped:
            coord_list, frame_list, face_land_marks = get_landmark_and_bbox(gt_img_list, 0)
            for bbox, frame, img_path in zip(coord_list, frame_list, gt_img_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                cv2.imwrite(img_path, crop_frame)

        print("handle generated_video_path")
        if get_file_type(generated_video_path)=="video":
            generated_input_basename = os.path.basename(generated_video_path).split('.')[0]
            save_generated_dir = os.path.join(os.path.join(result_save_path, "generated"), generated_input_basename)
            os.makedirs(save_generated_dir,exist_ok = True)
            print(save_generated_dir)
            cmd = f"ffmpeg -v fatal -i {generated_video_path} -start_number 0 {save_generated_dir}/%08d.png"
            os.system(cmd)
            generated_img_list = sorted(glob.glob(os.path.join(save_generated_dir, '*.[jpJP][pnPN]*[gG]')))
        elif get_file_type(generated_video_path)=="image":
            generated_img_list = [generated_video_path, ]
        elif os.path.isdir(generated_video_path):
            generated_img_list = glob.glob(os.path.join(generated_video_path, '*.[jpJP][pnPN]*[gG]'))
            generated_img_list = sorted(generated_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        else:
            print(f"{generated_video_path} should be a video file, an image file or a directory of images")
            continue

        if generated_cropped:
            coord_list, frame_list, face_land_marks = get_landmark_and_bbox(generated_img_list, 0)
            for bbox, frame, img_path in zip(coord_list, frame_list, generated_img_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                crop_frame = frame[y1:y2, x1:x2]
                crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
                cv2.imwrite(img_path, crop_frame)


        # align the ground truth and generated
        if len(gt_img_list) == 0 or len(generated_img_list)==0:
            print(f"video frames are zero, an image file or a directory of images")
            continue
        aligned_real_frames, aligned_generated_frames = align_images(gt_img_list, generated_img_list)
        results = metrics.evaluate_model(aligned_generated_frames, aligned_real_frames)
        psnr, ssim, cosine_similarity = results["visual_quality"]["avg_ssim_score"], results["visual_quality"]["avg_psnr_score"], results["identity_consistency"]["avg_cosine_similarity"]
        res_str += f"{task_id},{ssim},{psnr},{cosine_similarity}\n"
        ssims.append(ssim)
        psnrs.append(psnr)
        cosine_similaritys.append(cosine_similarity)
        # Define the file path
    # file_path = os.path.join(result_save_path, 'results.json')
    # # Save the results dictionary as a Pickle file
    # with open(file_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    avg_ssim = sum(ssims)/len(ssims)
    avg_psnr = sum(psnrs)/len(psnrs)
    avg_cosine_similarity = sum(cosine_similaritys)/len(cosine_similaritys)
    res_str += f"Average: {avg_ssim},{avg_psnr},{avg_cosine_similarity}"
    file_path = os.path.join(result_save_path, 'eval_results.txt')
    with open(file_path, 'w') as f:
        f.write(res_str)
        

    





if __name__ == "__main__":
    # print(f"main")
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate_config", type=str, default="configs/evaluate/test.yaml")
    # parser.add_argument("--result_dir", type=str, default="./evaluate_result")
    # parser.add_argument("--evaluate_type", default='clearness', choices=['clearness', 'identity', 'fid', 'lip_sync'], help="how to evaluate the images" ) 
    args = parser.parse_args()
    # print(f"=====main======")
    main(args)
