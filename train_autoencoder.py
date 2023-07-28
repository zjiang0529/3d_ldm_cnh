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

from utils import KL_loss, setup_ddp, prepare_json_dataloader, \
    distributed_all_gather, KL_loss_or, KL_loss_or_mean, NLayerDiscriminator3D, weights_init, hinge_d_loss, \
    PerceptualLossL1, Decoder
from visualize_image import visualize_one_slice_in_3d_image
from torch.cuda.amp import GradScaler, autocast


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

    def get_last_layer(self):
        return self.decoder.blocks[-1].conv.weight

    def calculate_adaptive_weight(self, recon_loss, g_loss, discriminator_weight=1.0):
        recon_grads = torch.autograd.grad(recon_loss, self.get_last_layer(), retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, self.get_last_layer(), retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight


def define_autoencoder(args, device=None):
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
    )
    data = next(iter(train_loader))
    print("Batch shape:", data["image"].shape)
    # Step 2: Define Autoencoder KL network and discriminator
    # autoencoder = define_instance(args, "autoencoder_def").to(device)
    autoencoder = define_autoencoder(args).to(device)
    num_params = 0
    for param in autoencoder.parameters():
        num_params += param.numel()
    print('[Autoencoder] Total number of parameters : %.3f M' % (num_params / 1e6))

    discriminator_norm = "BATCH"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        num_channels=32,
        in_channels=4,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    num_params = 0
    for param in discriminator.parameters():
        num_params += param.numel()
    print('[PatchDiscriminator] Total number of parameters : %.3f M' % (num_params / 1e6))

    args.model_dir = os.path.join(args.model_dir, args.exp_name)
    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
    trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if rank == 0:
        Path(os.path.join(args.model_dir, args.exp_name)).mkdir(parents=True, exist_ok=True)

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
            print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
        except:
            print(f"Rank {rank}: Train autoencoder from scratch.")

        try:
            discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location))
            print(f"Rank {rank}: Load trained discriminator from {trained_d_path}")
        except:
            print(f"Rank {rank}: Train discriminator from scratch.")

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        if discriminator_norm == "BATCH":
            discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss(reduction="mean")
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")

    if args.amp:
        loss_perceptual = PerceptualLossL1(spatial_dims=3, network_type="squeeze", is_fake_3d=True,
                                           fake_3d_ratio=0.2).eval()
    else:
        loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True,
                                         fake_3d_ratio=0.2).eval()

    loss_perceptual.to(device)

    # weight for mean reduction for everything
    adv_weight = args.autoencoder_train["adv_weight"]
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    kl_weight = args.autoencoder_train["kl_weight"]

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["lr"], betas=(0.5, 0.9),
                                   eps=1e-06 if args.amp else 1e-08)

    def lambda_rule(epoch):
        if epoch == 0:
            return 0.1
        else:
            return 1.0

    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lambda_rule)

    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["lr"], betas=(0.5, 0.9),
                                   eps=1e-06 if args.amp else 1e-08)

    args.tfevent_path = os.path.join(args.tfevent_path, args.exp_name)
    # initialize tensorboard writer
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 4: training
    autoencoder_warm_up_n_epochs = args.autoencoder_train["warm_up_n_epochs"]
    n_epochs = args.autoencoder_train["n_epochs"]
    val_interval = args.autoencoder_train["val_interval"]

    best_val_recon_epoch_loss = 100.0
    total_step = 0
    prev_time = time.time()

    if args.amp:
        # test use mean reduction for everything
        scaler_g = GradScaler(init_scale=2. ** 8, growth_factor=1.5)
        scaler_d = GradScaler(init_scale=2. ** 8, growth_factor=1.5)

    for epoch in range(n_epochs):
        # train
        autoencoder.train()
        discriminator.train()
        # if ddp_bool:
        #     # if ddp, distribute data across n gpus
        #     train_loader.sampler.set_epoch(epoch)
        #     val_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            images = batch["image"].as_tensor().to(device)
            with autocast(enabled=args.amp):
                reconstruction, z_mu, z_sigma = autoencoder(images)

                recons_loss = intensity_loss(reconstruction.contiguous(), images.contiguous())

                # compute loss_perceptual for each modality
                p_loss = 0
                for c in range(4):
                    p_loss += loss_perceptual(reconstruction[:, c:c + 1, ...].contiguous(),
                                              images[:, c:c + 1, ...].contiguous())
                p_loss = p_loss / 4.0

                # test use mean reduction for everything
                nll_loss = recons_loss
                kl_loss = KL_loss(z_mu, z_sigma)

                loss_g = nll_loss + kl_weight * kl_loss + perceptual_weight * p_loss

                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    d_weight = torch.tensor(1.0).to(device)

                    loss_g = loss_g + d_weight * adv_weight * generator_loss

            if args.amp:
                scaler_g.scale(loss_g).backward()
                # scaler_g.unscale_(optimizer_g)
                # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                with autocast(enabled=args.amp):

                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1

                tensorboard_writer.add_scalar("train/train_recon_loss_iter", recons_loss.detach().cpu().mean().item(),
                                              total_step)
                tensorboard_writer.add_scalar("train/train_nll_loss_iter", nll_loss.detach().cpu().item(), total_step)
                tensorboard_writer.add_scalar("train/train_kl_loss_iter", kl_loss.detach().cpu().item(), total_step)
                tensorboard_writer.add_scalar("train/train_perceptual_loss_iter", p_loss.detach().cpu().item(),
                                              total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    tensorboard_writer.add_scalar("train/train_adv_g_loss_iter", generator_loss.detach().cpu().item(),
                                                  total_step)
                    tensorboard_writer.add_scalar("train/train_adv_d_loss_iter",
                                                  discriminator_loss.detach().cpu().item(),
                                                  total_step)
                    tensorboard_writer.add_scalar("train/train_logit_fake_iter",
                                                  logits_fake.detach().cpu().mean().item(),
                                                  total_step)
                    tensorboard_writer.add_scalar("train/train_logit_real_iter",
                                                  logits_real.detach().cpu().mean().item(),
                                                  total_step)
                    tensorboard_writer.add_scalar("train/d_weight",
                                                  d_weight.detach().cpu().item(),
                                                  total_step)
                # Print log
                # Determine approximate time left
                batches_done = step
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %f] [D loss: %04f] [G loss: %01f, pixel: %04f, adv_g: %04f,"
                    " adv_d: %04f] ETA: %s"
                    % (
                        epoch,
                        n_epochs,
                        step,
                        len(train_loader),
                        scheduler_g.get_last_lr()[0],  # optimizer_g.param_groups[0]['lr'],
                        loss_d.detach().cpu().item() if epoch > autoencoder_warm_up_n_epochs else 0.0,
                        loss_g.detach().cpu().item(),
                        recons_loss.detach().cpu().mean().item(),
                        generator_loss.detach().cpu().item() if epoch > autoencoder_warm_up_n_epochs else 0.0,
                        discriminator_loss.detach().cpu().item() if epoch > autoencoder_warm_up_n_epochs else 0.0,
                        time_left,
                    )
                )
        scheduler_g.step()
        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            val_p_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    with autocast(enabled=args.amp):
                        reconstruction, z_mu, z_sigma = autoencoder(images)
                        recons_loss = intensity_loss(reconstruction.contiguous(), images.contiguous())
                        # compute loss_perceptual for each modality
                        p_loss = 0
                        for c in range(4):
                            p_loss += loss_perceptual(reconstruction[:, c:c + 1, ...].contiguous(),
                                                      images[:, c:c + 1, ...].contiguous())
                        p_loss = p_loss / 4.0
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
                print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")
                print(f"Epoch {epoch} val_p_loss: {val_p_epoch_loss}")
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if (val_recon_epoch_loss + perceptual_weight * val_p_epoch_loss) < best_val_recon_epoch_loss \
                        and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss + perceptual_weight * val_p_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                tensorboard_writer.add_scalar("val/val_recon_loss", val_recon_epoch_loss, total_step)
                tensorboard_writer.add_scalar("val/val_p_loss", val_p_epoch_loss, total_step)
                tensorboard_writer.add_scalar("val/val_total_loss",
                                              val_recon_epoch_loss + perceptual_weight * val_p_epoch_loss, total_step)
                for channel in range(4):
                    for axis in range(3):
                        tensorboard_writer.add_image(
                            f"val/channel_{channel}/val_img_{axis}",
                            visualize_one_slice_in_3d_image(torch.flip(images[0, channel, ...].cpu(), [-3, -2, -1]),
                                                            axis).transpose([2, 1, 0]),
                            total_step,
                        )
                        tensorboard_writer.add_image(
                            f"val/channel_{channel}/val_recon_{axis}",
                            visualize_one_slice_in_3d_image(
                                torch.flip(reconstruction[0, channel, ...].cpu(), [-3, -2, -1]), axis).transpose(
                                [2, 1, 0]),
                            total_step,
                        )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
