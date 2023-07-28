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

import os
import sys
from pathlib import Path
from typing import Sequence
import time
from datetime import timedelta

import torch
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import PatchDiscriminator, AutoencoderKL
from generative.networks.nets.autoencoderkl import Encoder
from monai.config import print_config
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim import lr_scheduler

from utils import KL_loss, define_instance, prepare_dataloader, setup_ddp, prepare_json_dataloader, \
    distributed_all_gather, KL_loss_or, KL_loss_or_mean, NLayerDiscriminator3D, weights_init, hinge_d_loss, \
    PerceptualLossL1, Decoder
from visualize_image import visualize_one_slice_in_3d_image
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import SaveImage

import torch.distributed as dist


class AutoencoderKLCKModified(AutoencoderKL):
    """
    Override encoder to make it align with original ldm codebase and support activation checkpointing.
    """

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int = 1,
            out_channels: int = 1,
            num_res_blocks: Sequence[int] = (2, 2, 2, 2),
            num_channels: Sequence[int] = (32, 64, 64, 64),
            attention_levels: Sequence[bool] = (False, False, True, True),
            latent_channels: int = 3,
            norm_num_groups: int = 32,
            norm_eps: float = 1e-6,
            with_encoder_nonlocal_attn: bool = True,
            with_decoder_nonlocal_attn: bool = True,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, num_res_blocks, num_channels, attention_levels,
                         latent_channels, norm_num_groups, norm_eps, with_encoder_nonlocal_attn,
                         with_decoder_nonlocal_attn, use_flash_attention)

        # Override decoder using transposed conv
        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            num_channels=num_channels,
            in_channels=latent_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_nonlocal_attn=with_decoder_nonlocal_attn,
            use_flash_attention=use_flash_attention,
        )

    def encode(self, x: torch.Tensor):
        """
        Forwards an image through the spatial encoder, obtaining the latent mean and sigma representations.

        Args:
            x: BxCx[SPATIAL DIMS] tensor

        """
        # h = self.encoder(x)
        h = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)

        z_mu = self.quant_conv_mu(h)
        z_log_var = self.quant_conv_log_sigma(h)
        z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
        z_sigma = torch.exp(z_log_var / 2)

        return z_mu, z_sigma

    def decode(self, z: torch.Tensor):
        """
        Based on a latent space sample, forwards it through the Decoder.

        Args:
            z: Bx[Z_CHANNELS]x[LATENT SPACE SHAPE]

        Returns:
            decoded image tensor
        """
        z = self.post_quant_conv(z)
        dec = torch.utils.checkpoint.checkpoint(self.decoder, z, use_reentrant=False)

        return dec

    def custom(self, module):
        def custom_forward(*inputs):
            inputs = module(inputs[0])
            return inputs

        return custom_forward


def define_autoencoder(args, device=None):
    # autoencoder = AutoencoderKL(
    # autoencoder = AutoencoderKL_CK(
    autoencoder = AutoencoderKLCKModified(
        spatial_dims=args.spatial_dims,
        in_channels=args.image_channels,
        out_channels=args.image_channels,
        num_channels=args.autoencoder_def["num_channels"],
        latent_channels=args.latent_channels,
        num_res_blocks=args.autoencoder_def["num_res_blocks"],
        norm_num_groups=args.autoencoder_def["norm_num_groups"],
        attention_levels=args.autoencoder_def["attention_levels"],
        with_encoder_nonlocal_attn=args.autoencoder_def["with_encoder_nonlocal_attn"],
        # current attention block causes stride warning when using ddp
        with_decoder_nonlocal_attn=args.autoencoder_def["with_decoder_nonlocal_attn"],
        use_flash_attention=args.autoencoder_def["use_flash_attention"]
    )

    return autoencoder


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
        default=False,
        action="store_true",
        help="whether to use AMP",
    )
    parser.add_argument(
        "-p",
        "--ckpt",
        default="autoencoder_last.pt",
        help="checkpoint name",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

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

    # Step 1: set data loader
    train_loader, val_loader = prepare_json_dataloader(
        args,
        args.autoencoder_train["batch_size"],
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=args.download_data,
        is_inference=True
    )
    data = next(iter(val_loader))
    print("Batch shape:", data["image"].shape)

    autoencoder = define_autoencoder(args).to(device)
    num_params = 0
    for param in autoencoder.parameters():
        num_params += param.numel()
    print('[Autoencoder] Total number of parameters : %.3f M' % (num_params / 1e6))

    args.model_dir = os.path.join(args.model_dir, args.exp_name)
    trained_g_path = os.path.join(args.model_dir, args.ckpt)

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    if args.amp:
        loss_perceptual = PerceptualLossL1(spatial_dims=3, network_type="squeeze", is_fake_3d=True,
                                           fake_3d_ratio=0.2).eval()
    else:
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True,
                                         fake_3d_ratio=0.2).eval()

    loss_perceptual.to(device)

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        # intensity_loss = L1Loss(reduction="none")
        intensity_loss = L1Loss(reduction="mean")
        if rank == 0:
            print("Use l1 loss")

    autoencoder.eval()
    val_recon_epoch_loss = 0
    val_p_epoch_loss = 0
    for step, batch in enumerate(val_loader):
        images = batch["image"].to(device)  # choose only one of Brats channels
        with torch.no_grad():
            with autocast(enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                recons_loss = intensity_loss(reconstruction.contiguous(), images.contiguous())
                p_loss = loss_perceptual(reconstruction.contiguous(), images.contiguous())
                reconstruction = (torch.clip(reconstruction, -1., 1.) + 1.) / 2.
                SaveImage(output_dir=args.model_dir, output_postfix="image", resample=False)(
                    reconstruction[0], meta_data={
                        'filename_or_obj': batch["image_meta_dict"]['filename_or_obj'][0].replace(".nii.gz", "")})
        val_recon_epoch_loss += recons_loss.item()
        val_p_epoch_loss += p_loss.item()
    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    val_p_epoch_loss = val_p_epoch_loss / (step + 1)

    print(
        f"Rank: {rank}, L1 Loss: {val_recon_epoch_loss:.4f}, "
        f"p Loss: {val_p_epoch_loss:.4f}, N_batch: {step + 1} ")

    val_recon_epoch_loss = torch.tensor(val_recon_epoch_loss).cuda(rank)
    val_p_epoch_loss = torch.tensor(val_p_epoch_loss).cuda(rank)

    if world_size > 1:
        loss_list = distributed_all_gather([val_recon_epoch_loss], out_numpy=True,
                                           )
        val_recon_epoch_loss = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)

        loss_list = distributed_all_gather([val_p_epoch_loss], out_numpy=True,
                                           )
        val_p_epoch_loss = np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0)

    if rank == 0:
        # save last model
        print(f"val_recon_loss: {val_recon_epoch_loss}")
        print(f"val_p_loss: {val_p_epoch_loss}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
