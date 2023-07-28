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
        args.diffusion_train["batch_size"],
        rank=rank,
        world_size=world_size,
        cache=1.0,
        download=False,
        data_aug=True
    )

    # initialize tensorboard writer
    args.tfevent_path = os.path.join(args.tfevent_path, args.exp_name)
    if rank == 0:
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "diffusion")
        tensorboard_writer = SummaryWriter(tensorboard_path)

    # Step 2: Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_autoencoder(args).to(device)

    args.model_dir = os.path.join(args.model_dir, args.exp_name)
    trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")

    # Set up spatial scaler
    label_spatial_scaler = SpatialRescaler(n_stages=len(args.autoencoder_def["num_channels"]) - 1, method='nearest')

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location))
    print(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")
    autoencoder.eval()

    if rank == 0:
        num_params = 0
        for param in autoencoder.parameters():
            num_params += param.numel()
        print('[Autoencoder] Total number of parameters : %.3f M' % (num_params / 1e6))

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast(enabled=True):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"][0:1, ...].to(device))
            if rank == 0:
                recon_img = autoencoder.decode_stage_2_outputs(z)
                print(f"Latent feature shape {z.shape}")
                for axis in range(3):
                    tensorboard_writer.add_image(
                        "train/train_img_" + str(axis),
                        visualize_one_slice_in_3d_image(torch.flip(check_data["image"][0, 0, ...], [-2, -1]),
                                                        axis).transpose([2, 1, 0]),
                        1,
                    )
                    tensorboard_writer.add_image(
                        "train/recon_img_" + str(axis),
                        visualize_one_slice_in_3d_image(torch.flip(recon_img[0, 0, ...], [-2, -1]), axis).transpose(
                            [2, 1, 0]),
                        1,
                    )
                del recon_img

    scale_factor = 1. / z.flatten().std()
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    print(f"Rank {rank}: scale_factor -> {scale_factor}")

    # Define Diffusion Model
    # add one for the conditional seg map
    unet = define_instance(args, "diffusion_def").to(device)

    if rank == 0:
        num_params = 0
        for param in unet.parameters():
            num_params += param.numel()
        print('[Unet] Total number of parameters : %.3f M' % (num_params / 1e6))

    trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location))
            print(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            print(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=False
    )

    if ddp_bool:
        # autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=False)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # We also define the ddim inferer for fast eval
    scheduler_ddim = DDIMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
        clip_sample=False
    )

    scheduler_ddim.set_timesteps(num_inference_steps=250)
    inferer_ddim = LatentDiffusionInferer(scheduler_ddim, scale_factor=scale_factor)

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["lr"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[5000, 15000], gamma=0.1)

    # Step 4: training
    n_epochs = args.diffusion_train["n_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler()
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    prev_time = time.time()
    # used for label map visualization

    colorize = torch.cat([
        torch.ones(3, 1, 1, 1) * -1,
        torch.randn(3, 4, 1, 1)],
        1).to(device)

    for epoch in range(n_epochs):
        unet.train()
        epoch_loss_ = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            label = batch["label"].to(device)
            brain_fg = (images[:, 0:1, ...] > -1).float()
            label = label + brain_fg
            # scale the label to latent shape
            downsampled_seg_label = label_spatial_scaler(label)
            # normalize it to [-1, 1]
            downsampled_seg_label = (downsampled_seg_label / 4.) * 2. - 1.
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                # if ddp_bool:
                #     inferer_autoencoder = autoencoder.module
                # else:
                #     inferer_autoencoder = autoencoder
                with torch.no_grad():
                    latent = autoencoder.encode_stage_2_inputs(images) * scale_factor

                noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                latent_model_input = torch.cat([noisy_latent, downsampled_seg_label], dim=1)

                noise_pred = unet(x=latent_model_input, timesteps=timesteps)

                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                tensorboard_writer.add_scalar("train/train_diffusion_loss_iter", loss.detach().cpu().item(), total_step)
                batches_done = step
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [LR: %f] [loss: %04f] ETA: %s "
                    % (
                        epoch,
                        n_epochs,
                        step,
                        len(train_loader),
                        lr_scheduler.get_last_lr()[0],
                        loss.detach().cpu().item(),
                        time_left,
                    )
                )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if ddp_bool:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            tensorboard_writer.add_scalar("train/train_diffusion_loss_epoch", epoch_loss.cpu().item(), total_step)

        torch.cuda.empty_cache()

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            with torch.no_grad():
                # compute val loss
                for step, batch in enumerate(val_loader):
                    images = batch["image"].to(device)
                    label = batch["label"].to(device)
                    brain_fg = (images[:, 0:1, ...] > -1).float()
                    label = label + brain_fg
                    # scale the label to latent shape
                    downsampled_seg_label = label_spatial_scaler(label)
                    # normalize it to [-1, 1]
                    downsampled_seg_label = (downsampled_seg_label / 4.) * 2. - 1.

                    with autocast(enabled=True):
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        latent = autoencoder.encode_stage_2_inputs(images) * scale_factor

                        noisy_latent = scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
                        latent_model_input = torch.cat([noisy_latent, downsampled_seg_label], dim=1)

                        noise_pred = unet(x=latent_model_input, timesteps=timesteps)

                        val_loss = F.mse_loss(noise_pred, noise)
                    val_recon_epoch_loss += val_loss.item()
                val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

                print(f"Rank: {rank}, Val Loss: {val_recon_epoch_loss:.4f}, ")

                val_recon_epoch_loss = torch.tensor(val_recon_epoch_loss).to(device)

                if ddp_bool:
                    dist.barrier()
                    dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)

                # write val loss and save best model
                if rank == 0:
                    tensorboard_writer.add_scalar("val/val_diffusion_loss", val_recon_epoch_loss, total_step)
                    print(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                    # save last model
                    if ddp_bool:
                        torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                    else:
                        torch.save(unet.state_dict(), trained_diffusion_path_last)

                    # save best model
                    if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                        best_val_recon_epoch_loss = val_recon_epoch_loss
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path)
                        print("Got best val noise pred loss.")
                        print("Save trained latent diffusion model to", trained_diffusion_path)

                    # visualize synthesized image

                    if epoch % (
                            val_interval) == 0:  # time cost of synthesizing images is large (use ddim sampler)
                        with autocast(enabled=True):
                            # synthetic_images = inferer_ddim.sample(
                            #     input_noise=noise[0:1, ...],
                            #     autoencoder_model=inferer_autoencoder,
                            #     diffusion_model=unet,
                            #     scheduler=scheduler_ddim,
                            #     verbose=True
                            # )

                            latents = noise[0:1, ...]
                            downsampled_seg_label = downsampled_seg_label[0:1, ...]

                            for t in tqdm(scheduler_ddim.timesteps, ncols=110):
                                latent_model_input = torch.cat(
                                    [latents, downsampled_seg_label], dim=1)
                                noise_pred = unet(
                                    x=latent_model_input, timesteps=torch.Tensor((t,)).to(device)
                                )
                                latents, _ = scheduler_ddim.step(noise_pred, t, latents)

                            synthetic_images = autoencoder.decode_stage_2_outputs(latents / scale_factor)
                        for channel in range(4):
                            for axis in range(3):
                                tensorboard_writer.add_image(
                                    f"val_channel_{channel}/diff_synimg_{axis}",
                                    visualize_one_slice_in_3d_image(
                                        torch.flip(synthetic_images[0, channel, ...].cpu(), [-3, -2, -1]),
                                        axis).transpose([2, 1, 0]),
                                    total_step,
                                )
                                tensorboard_writer.add_image(
                                    f"val_channel_{channel}/val_real_{axis}",
                                    visualize_one_slice_in_3d_image(
                                        torch.flip(images[0, channel, ...].cpu(), [-3, -2, -1]),
                                        axis).transpose(
                                        [2, 1, 0]),
                                    total_step,
                                )

                        for axis in range(3):
                            tensorboard_writer.add_image(
                                "val/diff_seg_" + str(axis),
                                visualize_one_slice_in_3d_label(colorize,
                                                                torch.flip(label[0:1, ...].to(device),
                                                                           [-3, -2, -1]), axis).transpose(
                                    [2, 1, 0]
                                ),
                                total_step,
                            )
                    del synthetic_images
                torch.cuda.empty_cache()
        lr_scheduler.step()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
