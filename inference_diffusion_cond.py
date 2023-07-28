# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
from pathlib import Path
import time
from datetime import timedelta
import warnings

import numpy as np

warnings.simplefilter('ignore', UserWarning)

import os
import sys

import torch
import torch.nn.functional as F
from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler
from monai.config import print_config
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from utils import define_instance, prepare_json_dataloader, setup_ddp, SpatialRescaler
from train_autoencoder import define_autoencoder
from visualize_image import visualize_one_slice_in_3d_image, visualize_one_slice_in_3d_label
from tqdm import tqdm
from monai.transforms import SaveImage
from PIL import Image
from monai.utils import set_determinism

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
)
from monai.data import DataLoader, CacheDataset, create_test_image_3d, partition_dataset


def main():
    parser = argparse.ArgumentParser(description="PyTorch VAE-GAN training")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_local.json",
        help="environment json file that stores environment path",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_32g_local.json",
        help="config json file that stores hyper-parameters",
    )
    parser.add_argument(
        "--download_data",
        default=False,
        action="store_true",
        help="whether to download Brats data before training",
    )
    parser.add_argument(
        "--amp",
        default=True,
        action="store_true",
        help="whether to use AMP",
    )
    parser.add_argument(
        "-a",
        "--ae_ckpt",
        default="autoencoder.pt",
        help="checkpoint name",
    )
    parser.add_argument(
        "-u",
        "--unet_ckpt",
        default="diffusion_unet_cond_last.pt",
        help="checkpoint name",
    )
    parser.add_argument(
        "-s",
        "--scale_factor",
        type=float,
        default=None,
        help="the scale of the latent space",
    )
    parser.add_argument(
        "--case_name",
        type=str,
        default=None,
        help="infer a single case",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="_no_kidney",
        help="prefix for inference case",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()
    set_determinism(43)
    # Step 0: configuration
    ddp_bool = args.gpus > 1  # whether to use distributed data parallel
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    prefix = ""
    # Step 1: set data loader
    if args.case_name is None:
        _, val_loader = prepare_json_dataloader(
            args,
            1,  # make batch_size=1
            rank=rank,
            world_size=world_size,
            cache=1.0,
            download=args.download_data,
            is_inference=True
        )
    else:
        prefix = args.prefix
        files = []
        str_img = os.path.join(args.data_base_dir, "images", args.case_name + ".nii.gz")
        str_seg = os.path.join(args.data_base_dir, args.case_name + "_label" + prefix + ".nii.gz")
        files.append({"image": str_img, "label": str_seg})
        common_transform = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200, b_min=-1, b_max=1, clip=True),
            EnsureTyped(keys="image", dtype=torch.float32),
        ]
        val_transforms = Compose(
            common_transform
        )
        val_ds = CacheDataset(
            data=files, transform=val_transforms, cache_rate=1.0, num_workers=1,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=False
        )

    data = next(iter(val_loader))
    print("Batch shape:", data["image"].shape)

    # Define AE
    autoencoder = define_autoencoder(args).to(device)

    # Set up spatial scaler
    label_spatial_scaler = SpatialRescaler(n_stages=len(args.autoencoder_def["num_channels"]) - 1, method='nearest')

    num_params = 0
    for param in autoencoder.parameters():
        num_params += param.numel()
    print('[Autoencoder] Total number of parameters : %.3f M' % (num_params / 1e6))

    args.model_dir = os.path.join(args.model_dir, args.exp_name)
    trained_ae_path = os.path.join(args.model_dir, args.ae_ckpt)

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_ae_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_ae_path}")

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(val_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"][0:1, ...].to(device))
            print(f"Latent feature shape {z.shape}")

    scale_factor = 1. / z.flatten().std()

    colorize = torch.cat([
        torch.ones(3, 1, 1, 1) * -1,
        torch.randn(3, 4, 1, 1)],
        1).to(device)

    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
        dist.all_reduce(colorize, op=torch.distributed.ReduceOp.AVG)

    if args.scale_factor is not None:
        scale_factor = args.scale_factor
    print(f"Rank {rank}: scale_factor -> {scale_factor}")

    # Define Unet
    unet = define_instance(args, "diffusion_def").to(device)
    trained_unet_path = os.path.join(args.model_dir, args.unet_ckpt)

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    unet.load_state_dict(torch.load(trained_unet_path, map_location=map_location))
    print(f"Rank {rank}: Load trained Unet from {trained_unet_path}")

    # We also define ddim inferer for fast eval
    scheduler_ddim = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=False
    )
    scheduler_ddim.set_timesteps(num_inference_steps=1000)

    autoencoder.eval()
    unet.eval()

    for step, batch in enumerate(val_loader):
        labels_ = batch["label"].to(device)  # choose only one of Brats channels
        images = batch["image"].to(device)
        with torch.no_grad():
            with autocast(enabled=args.amp):
                noise_shape = [1] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=labels_.dtype).to(device)
                brain_fg = (images[:, 0:1, ...] > -1).float()
                labels = labels_ + brain_fg
                # scale the label to latent shape
                downsampled_seg_label = label_spatial_scaler(labels)
                # normalize it to [-1, 1]
                downsampled_seg_label = (downsampled_seg_label / 4.) * 2. - 1.

                latents = noise
                for t in tqdm(scheduler_ddim.timesteps, ncols=110):
                    latent_model_input = torch.cat(
                        [latents, downsampled_seg_label], dim=1)
                    noise_pred = unet(
                        x=latent_model_input, timesteps=torch.Tensor((t,)).to(device)
                    )
                    latents, _ = scheduler_ddim.step(noise_pred, t, latents)

                synthetic_images = autoencoder.decode_stage_2_outputs(latents / scale_factor)
                synthetic_images = torch.clip(synthetic_images, -1., 1.)

                SaveImage(output_dir=args.model_dir, output_postfix="image", resample=False, separate_folder=False)(
                    (synthetic_images[0]+ 1.0) / 2.0, meta_data={
                        'filename_or_obj': batch["image_meta_dict"]['filename_or_obj'][0].replace(".nii.gz", "")})
                SaveImage(output_dir=args.model_dir, output_postfix="seg", resample=False, separate_folder=False)(
                        labels_[0], meta_data={
                        'filename_or_obj': batch["image_meta_dict"]['filename_or_obj'][0].replace(".nii.gz", "")})

                # synimg_list = []
                # seg_list = []
                # real_list = []
                # for axis in range(3):
                #     synimg = visualize_one_slice_in_3d_image(torch.flip(synthetic_images[0, 0, ...], [-3, -2, -1]),
                #                                         axis).transpose(
                #             [2, 1, 0]
                #         )
                #     synimg_list.append(synimg)
                #     seg = visualize_one_slice_in_3d_label(colorize,
                #                                         torch.flip(labels[0:1, ...].to(device),
                #                                                    [-3, -2, -1]), axis).transpose(
                #             [2, 1, 0]
                #         )
                #     seg_list.append(seg)
                #     real = visualize_one_slice_in_3d_image(torch.flip(images[0, 0, ...], [-3, -2, -1]),
                #                                         axis).transpose([2, 1, 0])
                #     real_list.append(real)
                #
                # file_name = batch["image_meta_dict"]['filename_or_obj'][0].split("/")[-1].split(".")[0]
                # seg_save = np.concatenate(seg_list,axis=2).transpose([1, 2, 0])
                # seg_save = Image.fromarray(seg_save)
                # seg_save.save(os.path.join(args.model_dir,file_name + prefix + "_seg.png"))
                #
                # synimg_save = np.concatenate(synimg_list, axis=2).transpose([1, 2, 0])
                # synimg_save = Image.fromarray(synimg_save)
                # synimg_save.save(os.path.join(args.model_dir, file_name + prefix + "_syn.png"))
                #
                # real_save = np.concatenate(real_list, axis=2).transpose([1, 2, 0])
                # real_save = Image.fromarray(real_save)
                # real_save.save(os.path.join(args.model_dir, file_name + prefix + "_real.png"))


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
