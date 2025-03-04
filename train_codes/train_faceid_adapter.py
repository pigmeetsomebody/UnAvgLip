import argparse
import itertools
import math
import os
import random
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt
import re
from copy import deepcopy
from models import SyncNet_color as SyncNet
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from insightface.app import FaceAnalysis
import time

RESIZED_IMG = 256
syncnet_T = 5
syncnet_mel_step_size = 16
face_model = "antelopev2"


def prepare_window(window):
    # 3 x T x H x W
    x = np.asarray(window) / 255.
    x = np.transpose(x, (3, 0, 1, 2))
    return x


logger = get_logger(__name__)

torch.cuda.empty_cache()


def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


logloss = nn.BCELoss()


def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss


def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]


def get_sync_loss(syncnet, mel, g, device):
    mel = mel.to(device)
    g = g.to(device)
    g = g[:, :, :, g.size(3) // 2:]
    g = torch.cat([g[:, i, :] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    print(f"a.shape: {a.shape}, v.shape: {v.shape}")
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)


def get_latest_checkpoint(output_dir):
    """
    Get the path to the latest checkpoint in the given output directory.

    Args:
        output_dir (str): Directory where checkpoints are stored.

    Returns:
        str: Path to the latest checkpoint, or None if no checkpoints are found.
    """
    # List all subdirectories in the output directory
    checkpoint_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    # Filter for directories matching the "checkpoint-{global_step}" pattern
    checkpoint_dirs = [d for d in checkpoint_dirs if re.match(r"checkpoint-\d+", d)]

    if not checkpoint_dirs:
        print("No checkpoints found in the directory.")
        return None

    # Sort by global step (extract the number from the directory name)
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)

    # Get the latest checkpoint
    latest_checkpoint = os.path.join(output_dir, checkpoint_dirs[0])
    print(f"Latest checkpoint found: {latest_checkpoint}")
    return latest_checkpoint


from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

import sys

sys.path.append("./")

from DataLoader import Dataset
from utils.utils import preprocess_img_tensor
from torch.utils import data as data_utils
from utils.model_utils import lora_validation, PositionalEncoding
import time
import pandas as pd
from PIL import Image
import torchvision.models as models

from identity_adapter.utils import is_torch2_available
from identity_adapter.identity_adapter import IPAdapterModule
from identity_adapter.utils import is_torch2_available, register_cross_attention_hook, get_net_attn_map, attnmaps2images, \
    concat_images

if is_torch2_available():
    from identity_adapter.attention_processor_faceid import LoRAIPAttnProcessor2_0 as LoRAIPAttnProcessor, \
        LoRAAttnProcessor2_0 as LoRAAttnProcessor
else:
    from identity_adapter.attention_processor_faceid import LoRAAttnProcessor, LoRAIPAttnProcessor

from identity_adapter.identity_perceiver import Resampler

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--unet_config_file",
        type=str,
        default=None,
        required=True,
        help="the configuration of unet file.",
    )
    parser.add_argument(
        "--reconstruction",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--testing_speed", action="store_true", help="Whether to caculate the running time")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--train_json", type=str, default="train.json",
                        help="The json file containing train image folders")
    parser.add_argument("--val_json", type=str, default="test.json",
                        help="The json file containing validation image folders")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint and are suitable for resuming training"
            " using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Conduct validation every X updates."
        ),
    )
    parser.add_argument(
        "--val_out_dir",
        type=str,
        default='',
        help=(
            "Conduct validation every X updates."
        ),
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--use_audio_length_left",
        type=int,
        default=1,
        help="number of audio length (left).",
    )
    parser.add_argument(
        "--use_audio_length_right",
        type=int,
        default=1,
        help="number of audio length (right)",
    )
    parser.add_argument(
        "--whisper_model_type",
        type=str,
        default="landmark_nearest",
        choices=["tiny", "largeV2"],
        help="Determine whisper feature type",
    )

    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )

    parser.add_argument('--syncnet_checkpoint_path', help='Load the pre-trained Expert discriminator', required=True,
                        type=str)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def print_model_dtypes(model, model_name):
    for name, param in model.named_parameters():
        if (param.dtype != torch.float32):
            print(f"{name}: {param.dtype}")


def perceptual_loss(vgg, pred_image, target_image):
    pred_features = vgg(pred_image)
    target_features = vgg(target_image)
    return F.l1_loss(pred_features, target_features)


# 边缘检测函数
def edge_detection_loss(pred_image, target_image, device):
    pred_image = pred_image.detach().cpu().numpy()
    # Normalize the image to range 0-255 and convert to uint8
    pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min()) * 255
    pred_image = pred_image.astype(np.uint8)
    pred_edges = cv2.Canny(pred_image, 100, 200)
    target_image = target_image.detach().cpu().numpy()
    target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min()) * 255
    target_image = target_image.astype(np.uint8)
    target_edges = cv2.Canny(target_image, 100, 200)
    pred_edges = torch.tensor(pred_edges).float().to(device)
    target_edges = torch.tensor(target_edges).float().to(device)
    return F.l1_loss(pred_edges, target_edges)


def main():
    args = parse_args()
    args.output_dir = f"output/{args.output_dir}"
    args.val_out_dir = f"val/{args.val_out_dir}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.val_out_dir, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')

    logging_dir = Path(args.output_dir, args.logging_dir)

    project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )

    print(f"Available GPUs: {torch.cuda.device_count()}")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=project_config,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # if args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
    #     raise ValueError(
    #         "Gradient accumulation is not supported when training the text encoder in distributed training. "
    #         "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
    #     )

    if args.seed is not None:
        #         set_seed(args.seed)
        set_seed(seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models and create wrapper for stable diffusion
    with open(args.unet_config_file, 'r') as f:
        unet_config = json.load(f)

    # text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    # Todo:
    print("Loading AutoencoderKL")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder='vae')
    # vae_fp32 = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    # print(f"Loading UNet2DConditionModel, unet_config: {unet_config}")

    model_path = "../models/musetalk/pytorch_model.bin"
    unet = UNet2DConditionModel(**unet_config)
    unet_weights = torch.load(model_path) if torch.cuda.is_available() else torch.load(model_path,
                                                                                       map_location=accelerator.device)
    unet.load_state_dict(unet_weights)
    unet.requires_grad_(False)
    # Freeze the weights in the copy
    print("Model Parameters:")
    total_params = 0
    for name, param in unet.named_parameters():
        print(f"{name}: {param.shape} | {param.numel()} parameters")
        total_params += param.numel()

    print(f"\n Unet Total parameters in model: {total_params}")
    time.sleep(5)

    for param in unet_weights.values():  # Assuming the weights are stored in a dictionary
        param.requires_grad = False

    # 使用预训练的VGG模型
    vgg = models.vgg19(pretrained=True).features[:16].eval().to(accelerator.device)
    for param in vgg.parameters():
        param.requires_grad = False

    face_app = FaceAnalysis(name=face_model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(RESIZED_IMG, RESIZED_IMG))

    # 初始化唇形同步判别器
    syncnet = SyncNet().to(accelerator.device)
    # 冻结syncnet的参数
    for p in syncnet.parameters():
        p.requires_grad = False
    # 加载唇形同步判别器
    print(f"load sync net: {args.syncnet_checkpoint_path}")
    load_checkpoint(args.syncnet_checkpoint_path, syncnet, None, reset_optimizer=True, overwrite_global_states=False)

    # init identity adapter
    weight_dtype = torch.float32

    image_proj_model = Resampler(
        dim=unet.config.cross_attention_dim,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=4,
        embedding_dim=512,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )

    # init adapter modules
    lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
            print(f"mid_block hidden_size: {hidden_size}")
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            print(f"up_blocks hidden_size: {hidden_size}")
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            print(f"down_blocks hidden_size: {hidden_size}")
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                 rank=lora_rank, lora_scale=1.0)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,
                                                   rank=lora_rank, lora_scale=1.0)
            attn_procs[name].load_state_dict(weights, strict=False)
    print(f"set unet atten processor: {attn_procs}")
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    total_params = 0

    for name, param in adapter_modules.named_parameters():
        print(f"{name}: {param.shape} | {param.numel()} parameters")
        total_params += param.numel()
    print(f"\n adapter_modules Total parameters in model: {total_params}")
    time.sleep(5)

    total_params = 0
    for name, param in image_proj_model.named_parameters():
        print(f"{name}: {param.shape} | {param.numel()} parameters")
        total_params += param.numel()

    print(f"\n image_proj_model Total parameters in model: {total_params}")
    time.sleep(5)
    for param in unet_weights.values():  # Assuming the weights are stored in a dictionary
        param.requires_grad = False

    vae.requires_grad_(False)
    # vae_fp32.requires_grad_(False)
    if args.whisper_model_type == "tiny":
        pe = PositionalEncoding(d_model=384)
    elif args.whisper_model_type == "largeV2":
        pe = PositionalEncoding(d_model=1280)
    else:
        print(f"not support whisper_model_type {args.whisper_model_type}")

    # weight_dtype = torch.float16
    # vae_fp32.to('cuda:4', dtype=weight_dtype)
    # vae_fp32.encoder = None
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    vae.to('cuda:1', dtype=weight_dtype)
    # vae.decoder = None
    pe.to(accelerator.device, dtype=weight_dtype)
    ip_adapter = IPAdapterModule(unet, image_proj_model, adapter_modules, accelerator.device, weight_dtype,
                                 args.pretrained_ip_adapter_path)

    print("Loading models done...")

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    # if args.scale_lr:
    #     args.learning_rate = (
    #         args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
    #     )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    print(f"loading train_dataset ...   train_batch_size: {args.train_batch_size}")
    train_dataset = Dataset(args.data_root,
                            args.train_json,
                            use_audio_length_left=args.use_audio_length_left,
                            use_audio_length_right=args.use_audio_length_right,
                            whisper_model_type=args.whisper_model_type
                            )
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        num_workers=8)
    print("loading val_dataset ...")
    val_dataset = Dataset(args.data_root,
                          args.val_json,
                          use_audio_length_left=args.use_audio_length_left,
                          use_audio_length_right=args.use_audio_length_right,
                          whisper_model_type=args.whisper_model_type
                          )
    val_data_loader = data_utils.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=8)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_data_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    ip_adapter, optimizer, train_data_loader, val_data_loader, lr_scheduler = accelerator.prepare(
        ip_adapter, optimizer, train_data_loader, val_data_loader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_data_loader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    print(f"  Num batches each epoch = {len(train_data_loader)}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_data_loader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Find the latest checkpoint
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None

        # path="../models/pytorch_model.bin"
        # TODO change path
        # path=None
    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path), strict=False)

        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

        # 初始化 TensorBoard 写入器
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.logging_dir)

        # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # caluate the elapsed time
    elapsed_time = []
    start = time.time()

    losses = []
    sync_losses = []
    visual_losses = []
    latent_losses = []
    lip_losses = []
    steps = []

    # 每100步记录loss
    log_interval = 100

    for epoch in range(first_epoch, args.num_train_epochs):
        # unet.train()
        for step, (ref_image, image, masked_image, masks, audio_feature, face_embedings, mels, kps) in enumerate(
                train_data_loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            dataloader_time = time.time() - start
            start = time.time()

            if mels.shape[0] <= 1 or image.shape[0] <= 1:
                print(f"invalid batchsize:{mels.shape[0]}")
                continue
            real_sync_loss = get_sync_loss(syncnet, mels, image, accelerator.device)
            print(f"real_sync_loss: {real_sync_loss.item()}")
            masks = masks.unsqueeze(1).to(vae.device)

            # print(f"g.shape: {g.shape}, mel: {mel.shape}, device: {device}")
            # for i in range(syncnet_T):
            #     print(f"ref_image[:, :, i].shape: {ref_image[:, i, :].shape}")
            ref_image = torch.cat([ref_image[:, i, :] for i in range(syncnet_T)], dim=0)
            image = torch.cat([image[:, i, :] for i in range(syncnet_T)], dim=0)
            masked_image = torch.cat([masked_image[:, i, :] for i in range(syncnet_T)], dim=0)
            audio_feature = torch.cat([audio_feature[:, i, :] for i in range(syncnet_T)], dim=0)
            face_embedings = torch.cat([face_embedings[:, i, :] for i in range(syncnet_T)], dim=0)

            print("=============epoch:{0}=step:{1}=====".format(epoch, step))
            print("ref_image: ", ref_image.shape)
            print("face_embedings: ", face_embedings.shape)
            print("masked_image: ", masked_image.shape)
            print("audio feature: ", audio_feature.shape)
            print("image: ", image.shape)

            ref_image = preprocess_img_tensor(ref_image).to(vae.device)
            image = preprocess_img_tensor(image).to(vae.device)
            masked_image = preprocess_img_tensor(masked_image).to(vae.device)
            face_embedings = face_embedings.to(vae.device)

            img_process_time = time.time() - start
            start = time.time()
            # 初始化一个空的loss列表和step列表

            with accelerator.accumulate(ip_adapter):
                vae = vae.half()
                # Convert images to latent space
                latents = vae.encode(image.to(dtype=weight_dtype)).latent_dist.sample()  # init image
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

                vae_time = time.time() - start
                start = time.time()

                # mask = torch.stack(
                #     [
                #         torch.nn.functional.interpolate(mask, size=(mask.shape[-1] // 8, mask.shape[-1] // 8))
                #         for mask in masks
                #     ]
                # )
                # mask = mask.reshape(-1, 1, mask.shape[-1], mask.shape[-1])

                bsz = latents.shape[0]
                # fix timestep for each image
                timesteps = torch.tensor([0], device=latents.device)
                # concatenate the latents with the mask and the masked latents
                """
                print("=============vae latents=====".format(epoch,step))
                print("ref_latents: ",ref_latents.shape)
                print("mask: ", mask.shape)
                print("masked_latents: ", masked_latents.shape)
                """

                # if unet_config['in_channels'] == 9:
                #     latent_model_input = torch.cat([mask, masked_latents, ref_latents], dim=1)
                # else:
                latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)

                audio_feature = audio_feature.to(dtype=weight_dtype)
                audio_feature = pe(audio_feature)

                # Predict the noise residual
                latent_model_input = latent_model_input.to(ip_adapter.device)
                timesteps = timesteps.to(ip_adapter.device)
                audio_feature = audio_feature.to(ip_adapter.device)
                face_embedings = face_embedings.to(ip_adapter.device)
                image_pred = ip_adapter(latent_model_input,
                                        timesteps,
                                        encoder_hidden_states=audio_feature, image_embeds=face_embedings)

                print(f"mels.shape: {mels.shape}")
                if args.reconstruction:  # decode the image from the predicted latents
                    image_pred = image_pred.to(vae.device)
                    image_pred_img = (1 / vae.config.scaling_factor) * image_pred
                    image_pred_img = image_pred_img.to(vae.device)
                    image_pred_img = vae.decode(image_pred_img.to(vae.dtype)).sample

                    image_pred_img = image_pred_img.to(accelerator.device)
                    image_pred = image_pred.to(accelerator.device)
                    latents = image_pred.to(accelerator.device)
                    # 计算唇形同步器损失
                    # mels = mels.squeeze(0)
                    # image_pred_img = image_pred_img.unsqueeze(0)

                    # image_pred_img_swapped = prepare_window(image_pred_img.detach().cpu().numpy())
                    # image_pred_img_swapped = image_pred_img.transpose()
                    t, c, h, w = image_pred_img.shape

                    selected_img = image_pred_img[0, :, :, :]
                    print(f"selected_img.shape: {selected_img.shape}")
                    selected_img_np = selected_img.detach().cpu().numpy()
                    selected_img_np = selected_img_np.transpose(1, 2, 0)
                    # Step 3: Ensure the image is in uint8 format
                    selected_img_np = (selected_img_np * 255).astype(np.uint8)
                    selected_faces_info = face_app.get(cv2.cvtColor(np.array(selected_img_np), cv2.COLOR_RGB2BGR))
                    if len(selected_faces_info) == 0:
                        continue
                    selected_face_info = selected_faces_info[-1]
                    selected_face_emb = selected_face_info['embedding']
                    target_face_emb = face_embedings[0, :, :]
                    if isinstance(selected_face_emb, np.ndarray):
                        selected_face_emb = torch.from_numpy(selected_face_emb)
                    selected_face_emb = selected_face_emb.to(target_face_emb.device)
                    identity_loss = F.l1_loss(selected_face_emb, target_face_emb, reduction='mean')
                    print(
                        f"selected_img: {selected_img.shape}, selected_face_emb: {selected_face_emb.shape}, target_face_emb: {target_face_emb.shape}, face_embedings.shape:{face_embedings.shape}, identity_distance:{identity_loss.item()}")

                    image_pred_img = image_pred_img.reshape(int(t / syncnet_T), syncnet_T, c, h, w)
                    image_pred_img = image_pred_img.to(torch.float32)
                    # image_pred_img = torch.FloatTensor(image_pred_img.cpu())

                    # mels = mels.squeeze()
                    # mels = mels.unsqueeze(0)
                    sync_loss = get_sync_loss(syncnet, mels, image_pred_img, accelerator.device)
                    if sync_loss.item() > 1:
                        continue
                    print(f"sync_loss:{sync_loss.item()}")

                    # image_pred_img = image_pred_img.squeeze(0)
                    # Mask the top half of the image and calculate the loss only for the lower half of the image.
                    image_pred_img = torch.cat([image_pred_img[:, i, :] for i in range(syncnet_T)], dim=0)
                    image_pred_img = image_pred_img[:, :, image_pred_img.shape[2] // 2:, :]

                    image = image[:, :, image.shape[2] // 2:, :]
                    print(f"image_pred_img.shape:{image_pred_img.shape}, image.shape: {image.shape}")
                    image_pred_img = image_pred_img.to(accelerator.device)
                    image = image.to(accelerator.device)

                    loss_perceptual = perceptual_loss(vgg, image_pred_img.float(), image.float())

                    identity_loss = identity_loss.to(accelerator.device)
                    loss_perceptual = loss_perceptual.to(accelerator.device)
                    sync_loss = sync_loss.to(accelerator.device)
                    loss = identity_loss + loss_perceptual + sync_loss
                    # loss =  loss_perceptual + sync_loss
                    print(
                        f"loss: {loss.item()}, identity_loss: {identity_loss.item()}, loss_perceptual: {loss_perceptual.item()}, sync_loss:{sync_loss.item()}")
                    # print(f"loss: {loss.item()}, loss_perceptual: {loss_perceptual.item()}, sync_loss:{sync_loss.item()}")

                    steps.append(global_step)
                    losses.append(loss.item())
                    lip_losses.append(identity_loss.item())
                    # latent_losses.append(loss_latents.item())
                    visual_losses.append(loss_perceptual.item())
                    sync_losses.append(sync_loss.item())
                else:
                    loss = F.mse_loss(image_pred.float(), latents.float(), reduction="mean")
                #

                unet_elapsed_time = time.time() - start
                start = time.time()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # 解封装模型
                    original_ip_adapter = accelerator.unwrap_model(ip_adapter)
                    params_to_clip = (
                        itertools.chain(original_ip_adapter.image_proj_model.parameters(),
                                        original_ip_adapter.adapter_modules.parameters())
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                backward_elapsed_time = time.time() - start
                start = time.time()

                if args.testing_speed is True and accelerator.is_main_process:
                    elapsed_time.append(
                        [dataloader_time, unet_elapsed_time, vae_time, backward_elapsed_time, img_process_time]
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path, safe_serialization=False)
                        logger.info(f"Saved state to {save_path}")
                        plt.figure(figsize=(10, 7))
                        plt.plot(steps, lip_losses, label="Identity Loss")
                        # plt.plot(steps, latent_losses, label="Latents Loss")
                        plt.plot(steps, visual_losses, label="Perceptual Loss")
                        plt.plot(steps, sync_losses, label="Sync Loss")
                        plt.plot(steps, losses, label="Total Loss")
                        plt.xlabel("Training Steps")
                        plt.ylabel("Loss")
                        plt.title("Training Loss Curves")
                        plt.legend()
                        # writer.add_figure("Loss Curves", plt.gcf(), global_step)
                        loss_curve_path = os.path.join(args.val_out_dir, f"loss_curve_step_{global_step}.png")
                        plt.savefig(loss_curve_path)
                        print(f"Loss curve saved to: {loss_curve_path}")
                        # 可选：保存图像文件
                        logger.info(f"Saved samples to images/val")
                        plt.close()
                        sync_data = pd.DataFrame({'x': steps, 'y': sync_losses})
                        # 将 DataFrame 保存为 CSV 文件
                        loss_curve_path = os.path.join(args.val_out_dir, f"sync_loss_curve_step_{global_step}.csv")
                        sync_data.to_csv(loss_curve_path, index=False)  # index=False 避免保存行索引

                        # latents_data = pd.DataFrame({'x': steps, 'y': latent_losses})
                        # latents_curve_path = os.path.join(args.val_out_dir, f"Latents_loss_curve_step_{global_step}.csv")
                        # latents_data.to_csv(latents_curve_path, index=False)  # index=False 避免保存行索引

                        percep_data = pd.DataFrame({'x': steps, 'y': visual_losses})
                        percep_curve_path = os.path.join(args.val_out_dir, f"percep_loss_curve_step_{global_step}.csv")
                        percep_data.to_csv(percep_curve_path, index=False)  # index=False 避免保存行索引

                        lip_losses_data = pd.DataFrame({'x': steps, 'y': lip_losses})
                        lip_curve_path = os.path.join(args.val_out_dir, f"identity_loss_curve_step_{global_step}.csv")
                        lip_losses_data.to_csv(lip_curve_path, index=False)  # index=False 避免保存行索引

                        total_losses_data = pd.DataFrame({'x': steps, 'y': losses})
                        total_curve_path = os.path.join(args.val_out_dir, f"total_loss_curve_step_{global_step}.csv")
                        total_losses_data.to_csv(total_curve_path,
                                                 index=False)  # index=False 避免保存行索引

                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        logger.info(
                            f"Running validation... epoch={epoch}, global_step={global_step}"
                        )
                        print("===========start validation==========")
                        # Use the helper function to check the data types for each model
                        vae_new = vae.float()

                        print(f"weight_dtype: {weight_dtype}")
                        print(f"epoch type: {type(epoch)}")
                        print(f"global_step type: {type(global_step)}")
                        lora_validation(
                            # vae=accelerator.unwrap_model(vae),
                            vae=vae,
                            vae_fp32=vae,
                            unet_config=unet_config,
                            unet_weights=unet_weights,
                            # weight_dtype=weight_dtype,
                            weight_dtype=torch.float32,
                            epoch=epoch,
                            global_step=global_step,
                            val_data_loader=val_data_loader,
                            output_dir=args.val_out_dir,
                            whisper_model_type=args.whisper_model_type,
                            ip_adapter=ip_adapter,
                            syncnet_T=5
                        )
                        # attn_maps = get_net_attn_map((256, 256), batch_size=1)
                        # attn_hot = attnmaps2images(attn_maps)
                        # fig, axes = plt.subplots(1, len(attn_hot), figsize=(12, 4))
                        # for axe, image in zip(axes, attn_hot):
                        #     axe.imshow(image, cmap='gray')
                        #     axe.axis('off')
                        # val_img_dir = f"images/{args.val_out_dir}/{global_step}"
                        # if not os.path.exists(val_img_dir):
                        #     os.makedirs(val_img_dir)
                        # atten_map_img_save_path = '{0}/attn_map_epoch_{1}_{2}_image.png'.format(val_img_dir, global_step,step)
                        # print("valtion in step:{0}, time:{1}. attention_map_saved:\n{2}".format(step,time.time()-start, atten_map_img_save_path))

                    start = time.time()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                    "unet": unet_elapsed_time,
                    "backward": backward_elapsed_time,
                    "data": dataloader_time,
                    "img_process": img_process_time,
                    "vae": vae_time
                    }
            progress_bar.set_postfix(**logs)
            #             accelerator.log(logs, step=global_step)

            accelerator.log(
                {
                    "loss/step_loss": logs["loss"],
                    "parameter/lr": logs["lr"],
                    "time/unet_forward_time": unet_elapsed_time,
                    "time/unet_backward_time": backward_elapsed_time,
                    "time/data_time": dataloader_time,
                    "time/img_process_time": img_process_time,
                    "time/vae_time": vae_time
                },
                step=global_step,
            )

            torch.cuda.empty_cache()
            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer.close()
    accelerator.end_training()


if __name__ == "__main__":
    main()