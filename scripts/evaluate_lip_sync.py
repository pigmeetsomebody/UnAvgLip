import torch
import numpy as np
from pytorch_msssim import ssim
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import euclidean
import sys
sys.path.append('./Wav2Lip')  # Path to Wav2Lip repository
from models.syncnet import SyncNet

class FaceGenerationMetrics:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.mtcnn = MTCNN(device=device)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.syncnet = SyncNet().to(device)
        # Load pretrained syncnet
        checkpoint = torch.load('./Wav2Lip/checkpoints/syncnet_checkpoint.pth')
        self.syncnet.load_state_dict(checkpoint['state_dict'])
        self.syncnet.eval()

    def compute_lip_sync_quality(self, video_frames, mel):
        """
        Args:
            video_frames: Tensor (T, 3, H, W)
            mel: Mel spectrogram tensor (1, 80, T)
        Returns:
            Dictionary with sync confidence scores
        """
        with torch.no_grad():
            audio_embeds = self.syncnet.audio_encoder(mel)
            video_embeds = self.syncnet.face_encoder(video_frames)
            
            # Cosine similarity
            audio_embeds = audio_embeds.view(audio_embeds.size(0), -1)
            video_embeds = video_embeds.view(video_embeds.size(0), -1)
            cosine_sim = F.cosine_similarity(audio_embeds, video_embeds)
            
            return {
                'sync_confidence': cosine_sim.mean().item(),
                'frame_wise_confidence': cosine_sim.cpu().numpy()
            }

    # Visual quality and identity consistency methods remain unchanged