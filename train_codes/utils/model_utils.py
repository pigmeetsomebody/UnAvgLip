import torch
import torch.nn as nn

import torch
import torch.nn as nn
import time
import math
from .utils import decode_latents, preprocess_img_tensor
import os
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from torch import Tensor, nn
import logging
import json
import cv2
import numpy as np

from ip_adapter.ip_adapter_faceid import MLPProjModel
from ip_adapter.utils import is_torch2_available
from ip_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor
from ip_adapter.ip_adapter import IPAdapterModule
from ip_adapter.ip_adapter_faceid_separate import IPAttnProcessor, AttnProcessor, IPAdapterFaceID

RESIZED_IMG = 256

def convert_to_edge_image(pred_image, weight_dtype):
    pred_image_np = np.array(pred_image)
    pred_image_ts = torch.tensor(pred_image_np, dtype=weight_dtype)
    # Move to CPU and convert to NumPy array
    pred_image_np = pred_image_ts.cpu().detach().numpy()

    # Ensure pred_image_np is in the correct format for Canny
    pred_image_np = (pred_image_np - pred_image_np.min()) / (pred_image_np.max() - pred_image_np.min()) * 255
    pred_image_np = pred_image_np.astype(np.uint8)

    # Convert to grayscale if necessary
    if len(pred_image_np.shape) == 3 and pred_image_np.shape[2] > 1:
        pred_image_np = cv2.cvtColor(pred_image_np, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    pred_edges = cv2.Canny(pred_image_np, 100, 200)
    pred_edges = pred_edges.astype(np.uint8)
    edge_image = Image.fromarray(pred_edges)
    return edge_image




class PositionalEncoding(nn.Module):
    """
    Transformer 中的位置编码（positional encoding）
    """
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        #print(b, seq_len, d_model)
        x = x + pe.to(x.device)
        return x



def lora_validation(vae: torch.nn.Module,
               vae_fp32: torch.nn.Module,
      unet_config,
      unet_weights,
      weight_dtype: torch.dtype,
      epoch: int,
      global_step: int,
      val_data_loader,
      output_dir,
      whisper_model_type,
      ip_adapter: IPAdapterFaceID,
      UNet2DConditionModel=UNet2DConditionModel,
      syncnet_T=1,
     ):
    #print(f"unet copy: {unet_copy}")
     # Get the validation pipeline

    unet_copy = UNet2DConditionModel(**unet_config)
    unet_copy.load_state_dict(unet_weights)
    unet_copy.requires_grad_(False)

    # unet_copy.load_state_dict(unet.state_dict())
    
    #ip-adapter
    # image_proj_model = MLPProjModel(
    #     cross_attention_dim=unet.config.cross_attention_dim,
    #     id_embeddings_dim=512,
    #     num_tokens=4,
    # )
    
    

    # init adapter modules
    # lora_rank = 128
    # attn_procs = {}

    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #         # print(f"mid_block hidden_size: {hidden_size}")
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #         # print(f"up_blocks hidden_size: {hidden_size}")
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #         # print(f"down_blocks hidden_size: {hidden_size}")
    #     if cross_attention_dim is None:
    #         attn_procs[name] = AttnProcessor()
    #     else:
    #         layer_name = name.split(".processor")[0]
    #         weights = {
    #             "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
    #             "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
    #         }
    #         attn_procs[name] = IPAttnProcessor(
    #                 hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0, num_tokens=50,
    #             ).to(vae.device, dtype=weight_dtype)
    #         attn_procs[name].load_state_dict(weights, strict=False)
    # unet_copy.set_attn_processor(attn_procs)

    # attn_procs = {}
    # for name in unet_copy.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet_copy.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet_copy.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet_copy.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet_copy.config.block_out_channels[block_id]
    #     if cross_attention_dim is None:
    #         attn_procs[name] = AttnProcessor()
    #     else:
    #         attn_procs[name] = IPAttnProcessor(
    #             hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=0.75, num_tokens=50,
    #         ).to(vae.device, dtype=weight_dtype)
        
    #     unet_copy.set_attn_processor(attn_procs)

    # unet_copy.load_state_dict(unet_sd)
    unet_copy.to(vae.device).to(dtype=weight_dtype)
    
    ip_adapter.eval()
    unet_copy.eval()
    
    if whisper_model_type == "tiny":
        pe = PositionalEncoding(d_model=384)
    elif whisper_model_type == "largeV2":
        pe = PositionalEncoding(d_model=1280)
    elif whisper_model_type == "tiny-conv":
        pe = PositionalEncoding(d_model=384)
        # print(f" whisper_model_type: {whisper_model_type} Validation does not need PE")
    else:
        print(f"not support whisper_model_type {whisper_model_type}")
    pe.to(vae.device, dtype=weight_dtype)
    
    start = time.time()
    with torch.no_grad():
        for step, (ref_image, image, masked_image, masks, audio_feature, face_embedings, mel, kps) in enumerate(val_data_loader):


            print("ref_image: ",ref_image.shape)
            print("face_embedings: ", face_embedings.shape)
            print("masked_image: ", masked_image.shape)
            print("audio feature: ", audio_feature.shape)
            print("image: ", image.shape)
            print("======================")

            masks = masks.unsqueeze(1).unsqueeze(1).to(vae.device)
            ref_image = torch.cat([ref_image[:, i, :] for i in range(syncnet_T)], dim=0)
            image = torch.cat([image[:, i, :] for i in range(syncnet_T)], dim=0)
            masked_image = torch.cat([masked_image[:, i, :] for i in range(syncnet_T)], dim=0)
            audio_feature = torch.cat([audio_feature[:, i, :] for i in range(syncnet_T)], dim=0)
            face_embedings = torch.cat([face_embedings[:, i, :] for i in range(syncnet_T)], dim=0)

            print("ref_image: ",ref_image.shape)
            print("face_embedings: ", face_embedings.shape)
            print("masked_image: ", masked_image.shape)
            print("audio feature: ", audio_feature.shape)
            print("image: ", image.shape)
            
            ref_image = preprocess_img_tensor(ref_image).to(vae.device)
            image = preprocess_img_tensor(image).to(vae.device)
            masked_image = preprocess_img_tensor(masked_image).to(vae.device)
             # Convert images to latent space 
            latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample() # init image
            latents = latents * vae.config.scaling_factor
            # Convert masked images to latent space
            masked_latents = vae.encode(
                masked_image.reshape(image.shape).to(dtype=weight_dtype)  # masked image
            ).latent_dist.sample()
            masked_latents = masked_latents * vae.config.scaling_factor
            # Convert ref images to latent space
            ref_latents = vae.encode(
                ref_image.reshape(image.shape).to(dtype=weight_dtype)  # ref image
            ).latent_dist.sample()
            ref_latents = ref_latents * vae.config.scaling_factor

            # mask = torch.stack(
            #     [
            #         torch.nn.functional.interpolate(mask, size=(mask.shape[-1] // 8, mask.shape[-1] // 8))
            #         for mask in masks
            #     ]
            # )
            # mask = mask.reshape(-1, 1, mask.shape[-1], mask.shape[-1])

            bsz = latents.shape[0]
            timesteps = torch.tensor([0], device=latents.device)

            # if unet_config['in_channels'] == 9:
            #     latent_model_input = torch.cat([mask, masked_latents, ref_latents], dim=1)
            # else:
            #     latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
            latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

            # print(f"unet_copy: {unet_copy.device}")
            # print(f"latent_model_input: {latent_model_input.device}")
            # print(f"timesteps: {timesteps.device}")
            # print(f"audio_feature: {audio_feature.device}")
            # # print(f"ip_adapter_copy: {ip_adapter_copy.device}")
            # print(f"face_embedings: {face_embedings.device}")
            latent_model_input = latent_model_input.to(vae.device)
            timesteps = timesteps.to(vae.device)
            audio_feature = audio_feature.to(vae.device)
            image_pred = unet_copy(latent_model_input, timesteps, encoder_hidden_states = audio_feature).sample
            if ip_adapter is not None:
                latent_model_input = latent_model_input.to(ip_adapter.device)
                timesteps = timesteps.to(ip_adapter.device)
                audio_feature = audio_feature.to(ip_adapter.device)
                face_embedings = face_embedings.to(ip_adapter.device)
                adapter_image_pred = ip_adapter(latent_model_input, 
                                  timesteps, 
                                  encoder_hidden_states=audio_feature, image_embeds=face_embedings)
            else:
                adapter_image_pred = masked_latents

            print(f"adapter_image_pred.shape: {adapter_image_pred.shape} image_pred.shape: {image_pred.shape}")
            adapter_image_pred = adapter_image_pred.to(vae_fp32.device)
            latents = latents.to(vae_fp32.device)
            image_pred = image_pred.to(vae_fp32.device)
            ref_latents = ref_latents.to(vae_fp32.device)

            image = Image.new('RGB', (RESIZED_IMG*5, RESIZED_IMG*4))
            target_image = decode_latents(vae_fp32, latents)
            ref_image = decode_latents(vae_fp32, ref_latents)
            pred_image = decode_latents(vae_fp32, image_pred, ref_image)
            adapter_image = decode_latents(vae_fp32, adapter_image_pred, ref_image)
            # t, c, h, w = target_image.shape
            # target_image = target_image.reshape(int(t/syncnet_T), syncnet_T, c, h, w)
            # pred_image = pred_image.reshape(int(t/syncnet_T), syncnet_T, c, h, w)
            # adapter_image = adapter_image.reshape(int(t/syncnet_T), syncnet_T, c, h, w)
            # ref_image = ref_image.reshape(int(t/syncnet_T), syncnet_T, c, h, w)
            
            for i in range(0, syncnet_T):
                print(f"ref_image[{i}]: {ref_image[i]}")
                image.paste(ref_image[i], (i*RESIZED_IMG, 0 * RESIZED_IMG))

            for i in range(0, syncnet_T):
                image.paste(target_image[i], (i*RESIZED_IMG, 1 * RESIZED_IMG))
            
            for i in range(0, syncnet_T):
                image.paste(pred_image[i], (i*RESIZED_IMG, 2 * RESIZED_IMG))

            for i in range(0, syncnet_T):
                image.paste(adapter_image[i], (i*RESIZED_IMG, 3 * RESIZED_IMG))

            
            # image.paste(ref_image, (0, 0))
            # image.paste(target_image, (RESIZED_IMG, 0))
            # image.paste(pred_image, (RESIZED_IMG*2, 0))
            # image.paste(adapter_image, (RESIZED_IMG*3, 0))
            # pred_image_edge_image = convert_to_edge_image(pred_image, weight_dtype)
            # adapter_image_pred_edge_image = convert_to_edge_image(adapter_image, weight_dtype)

            # pred_edges_np = torch.tensor(pred_image).cpu().detach().numpy()
            # pred_image_np = (pred_image_np - pred_image_np.min()) / (pred_image_np.max() - pred_image_np.min()) * 255
            # pred_image_np = pred_image_np.astype(np.uint8)

            # if len(pred_image_np.shape) == 3 and pred_image_np.shape[2] > 1:
            #     pred_image_np = cv2.cvtColor(pred_image_np, cv2.COLOR_BGR2GRAY)
            # pred_edges = cv2.Canny(pred_image_np, 100, 200)
            # pred_edges_np = pred_edges.numpy()
            # pred_edges_img = Image.fromarray((pred_edges_np * 255).astype(np.uint8))
            
            # taget_edges = cv2.Canny(target_image.cpu().numpy(), 100, 200)
            # taget_edges_np = taget_edges.numpy()
            # taget_edges_image = Image.fromarray((taget_edges_np * 255).astype(np.uint8))

            # image.paste(pred_image_edge_image, (RESIZED_IMG*4, 0))
            # image.paste(adapter_image_pred_edge_image, (RESIZED_IMG*5, 0))

            val_img_dir = f"images/{output_dir}/{global_step}"
            val_gt_img_dir = f"images/{output_dir}/gt_img"
            val_adapter_img_dir = f"images/{output_dir}/adapter_img"
            val_ori_img_dir = f"images/{output_dir}/non_adapter_img"
            if not os.path.exists(val_gt_img_dir):
                os.makedirs(val_gt_img_dir)
            if not os.path.exists(val_adapter_img_dir):
                os.makedirs(val_adapter_img_dir)
            if not os.path.exists(val_ori_img_dir):
                os.makedirs(val_ori_img_dir)
            if not os.path.exists(val_img_dir):
                os.makedirs(val_img_dir)
            # target_image.save('{0}/val_epoch_{1}_{2}_image.png'.format(val_gt_img_dir, global_step,step))
            # pred_image.save('{0}/val_epoch_{1}_{2}_image.png'.format(val_ori_img_dir, global_step,step))
            # adapter_image.save('{0}/val_epoch_{1}_{2}_image.png'.format(val_adapter_img_dir, global_step,step))
            image.save('{0}/val_epoch_{1}_{2}_image.png'.format(val_img_dir, global_step,step))
            image_path = '{0}/val_epoch_{1}_{2}_image.png'.format(val_img_dir, global_step,step)
            print("valtion in step:{0}, time:{1}. image_saved:{2}".format(step,time.time()-start, image_path))

        print("valtion_done in epoch:{0}, time:{1}".format(epoch,time.time()-start))
    
