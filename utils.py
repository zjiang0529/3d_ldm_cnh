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

import os
import copy
from datetime import timedelta
from glob import glob
import json
import functools

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from monai.apps import DecathlonDataset
from monai.bundle import ConfigParser
from monai.data import DataLoader, CacheDataset, create_test_image_3d, partition_dataset
from monai.transforms import (
    AddChanneld,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    ScaleIntensityRangePercentilesd,
    Identityd
)
from typing import Sequence
from monai.networks.blocks import Convolution
from generative.networks.nets.autoencoderkl import AttentionBlock, ResBlock
from functools import partial


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def add_data_dir2path(list_files, data_dir):
    files = []
    for _i in range(len(list_files)):
        if isinstance(list_files[_i]["image"], list):
            str_img = [os.path.join(data_dir, list_files[_i]["image"][j]) for j in
                       range(len(list_files[_i]["image"], ))]
            str_seg = os.path.join(data_dir, list_files[_i]["label"])
            # for j in range(len(list_files[_i]["image"])):
            #     str_img = os.path.join(data_dir, list_files[_i]["image"][j])
            #     str_seg = os.path.join(data_dir, list_files[_i]["label"])
            #     files.append({"image": str_img, "label": str_seg})
            files.append({"image": str_img, "label": str_seg})
        else:
            str_img = os.path.join(data_dir, list_files[_i]["image"])
            str_seg = os.path.join(data_dir, list_files[_i]["label"])
            files.append({"image": str_img, "label": str_seg})
    return copy.deepcopy(files)

def add_data_dir2path_fold(json_data, data_dir, fold):
    list_train = []
    list_valid = []
    for item in json_data['training']:
        if item["fold"] == fold:
            item.pop("fold", None)
            list_valid.append(item)
        else:
            item.pop("fold", None)
            list_train.append(item)

    files = []
    for _i in range(len(list_train)):

        if isinstance(list_train[_i]["image"], list):
            str_img = [os.path.join(data_dir, list_train[_i]["image"][j]) for j in
                       range(len(list_train[_i]["image"], ))]
            str_seg = os.path.join(data_dir, list_train[_i]["label"])

        else:
            str_img = os.path.join(data_dir, list_train[_i]["image"])
            str_seg = os.path.join(data_dir, list_train[_i]["label"])

        # if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
        #     continue

        files.append({"image": str_img, "label": str_seg})

    train_files = copy.deepcopy(files)

    files = []
    for _i in range(len(list_valid)):
        if isinstance(list_valid[_i]["image"], list):
            str_img = [os.path.join(data_dir, list_valid[_i]["image"][j]) for j in
                       range(len(list_valid[_i]["image"], ))]
            str_seg = os.path.join(data_dir, list_valid[_i]["label"])
        else:
            str_img = os.path.join(data_dir, list_valid[_i]["image"])
            str_seg = os.path.join(data_dir, list_valid[_i]["label"])

        # if (not os.path.exists(str_img)) or (not os.path.exists(str_seg)):
        #     continue

        files.append({"image": str_img, "label": str_seg})
    val_files = copy.deepcopy(files)

    return train_files, val_files

def prepare_json_dataloader(
        args,
        batch_size,
        rank=0,
        world_size=1,
        cache=1.0,
        download=False,
        is_inference=False,
        data_aug=False,
        fold=0,
        load_label=True,
):
    ddp_bool = world_size > 1

    if isinstance(args.json_list, list):
        assert isinstance(args.data_base_dir, list)
        list_train = []
        list_valid = []
        for json_list, data_base_dir in zip(args.json_list, args.data_base_dir):
            with open(json_list, "r") as f:
                json_data = json.load(f)
            train, val = add_data_dir2path_fold(json_data, data_base_dir, fold=fold)
            list_train += train
            list_valid += val
    else:
        with open(args.json_list, "r") as f:
            json_data = json.load(f)

        list_train = add_data_dir2path(json_data['training'], args.data_base_dir)
        list_valid = add_data_dir2path(json_data['validation'], args.data_base_dir)

    print(f"Training files:{len(list_train)}, Val files:{len(list_valid)}")

    compute_dtype = torch.float32

    if load_label:
        keys = ["image", "label"]
    else:
        keys = ["image"]

    common_transform = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        CenterSpatialCropd(keys=keys, roi_size=args.roi_size),
        ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=-1, b_max=1,
                                        channel_wise=True, clip=True),
        EnsureTyped(keys="image", dtype=compute_dtype),
    ]
    random_transform = [
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
    ]

    if data_aug:
        train_transforms = Compose(
            common_transform + random_transform
        )
    else:
        train_transforms = Compose(
            common_transform
        )

    val_transforms = Compose(
        common_transform
    )
    if ddp_bool:
        list_train = partition_dataset(
            data=list_train,
            shuffle=True,
            num_partitions=world_size,
            even_divisible=True,
        )[rank]
        list_valid = partition_dataset(
            data=list_valid,
            shuffle=False,
            num_partitions=world_size,
            even_divisible=False,
        )[rank]

    train_loader = None

    if not is_inference:
        train_ds = CacheDataset(
            data=list_train, transform=train_transforms, cache_rate=cache, num_workers=8,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False
        )

    val_ds = CacheDataset(
        data=list_valid, transform=val_transforms, cache_rate=cache, num_workers=8,
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False
    )

    if rank == 0:
        print(f'Image shape {val_ds[0]["image"].shape}')
    return train_loader, val_loader


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def KL_loss_or(z_mu, z_sigma):
    # recover z_log_var
    z_log_var = torch.log(z_sigma) * 2
    var = torch.exp(z_log_var)
    kl_loss = 0.5 * torch.sum(
        torch.pow(z_mu, 2) + var - 1.0 - z_log_var, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def KL_loss_or_mean(z_mu, z_sigma):
    # recover z_log_var
    z_log_var = torch.log(z_sigma) * 2
    var = torch.exp(z_log_var)
    kl_loss = 0.5 * torch.mean(
        torch.pow(z_mu, 2) + var - 1.0 - z_log_var, dim=[1, 2, 3, 4])
    return kl_loss


def distributed_all_gather(
        tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix taking 3D input
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, False)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


from generative.losses import PerceptualLoss
from lpips_l1 import LPIPSL1


class PerceptualLossL1(PerceptualLoss):
    """
    Perceptual loss using features from pretrained deep neural networks trained. The function supports networks
    pretrained on: ImageNet that use the LPIPS approach from Zhang, et al. "The unreasonable effectiveness of deep
    features as a perceptual metric." https://arxiv.org/abs/1801.03924 ; RadImagenet from Mei, et al. "RadImageNet: An
    Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning"
    https://pubs.rsna.org/doi/full/10.1148/ryai.210315 ; and MedicalNet from Chen et al. "Med3D: Transfer Learning for
    3D Medical Image Analysis" https://arxiv.org/abs/1904.00625 .

    The fake 3D implementation is based on a 2.5D approach where we calculate the 2D perceptual on slices from the
    three axis.

    Args:
        spatial_dims: number of spatial dimensions.
        network_type: {``"alex"``, ``"vgg"``, ``"squeeze"``, ``"radimagenet_resnet50"``,
        ``"medicalnet_resnet10_23datasets"``, ``"medicalnet_resnet50_23datasets"``}
            Specifies the network architecture to use. Defaults to ``"alex"``.
        is_fake_3d: if True use 2.5D approach for a 3D perceptual loss.
        fake_3d_ratio: ratio of how many slices per axis are used in the 2.5D approach.
        cache_dir: path to cache directory to save the pretrained network weights.
    """

    def __init__(
            self,
            spatial_dims,
            network_type="alex",
            is_fake_3d=True,
            fake_3d_ratio=0.5,
            cache_dir=None,
    ):
        super().__init__(spatial_dims, network_type, is_fake_3d, fake_3d_ratio, cache_dir)
        self.perceptual_function = LPIPSL1(pretrained=True, net=network_type, verbose=False)


class Decoder(nn.Module):
    """
    Convolutional cascade upsampling from a spatial latent space into an image space.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        num_channels: sequence of block output channels.
        in_channels: number of channels in the bottom layer (latent space) of the autoencoder.
        out_channels: number of output channels.
        num_res_blocks: number of residual blocks (see ResBlock) per level.
        norm_num_groups: number of groups for the GroupNorm layers, num_channels must be divisible by this number.
        norm_eps: epsilon for the normalization.
        attention_levels: indicate which level from num_channels contain an attention block.
        with_nonlocal_attn: if True use non-local attention block.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
    """

    def __init__(
            self,
            spatial_dims: int,
            num_channels: Sequence[int],
            in_channels: int,
            out_channels: int,
            num_res_blocks: Sequence[int],
            norm_num_groups: int,
            norm_eps: float,
            attention_levels: Sequence[bool],
            with_nonlocal_attn: bool = True,
            use_flash_attention: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.norm_num_groups = norm_num_groups
        self.norm_eps = norm_eps
        self.attention_levels = attention_levels

        reversed_block_out_channels = list(reversed(num_channels))

        blocks = []
        # Initial convolution
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=reversed_block_out_channels[0],
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        # Non-local attention block
        if with_nonlocal_attn is True:
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )
            blocks.append(
                AttentionBlock(
                    spatial_dims=spatial_dims,
                    num_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    use_flash_attention=use_flash_attention,
                )
            )
            blocks.append(
                ResBlock(
                    spatial_dims=spatial_dims,
                    in_channels=reversed_block_out_channels[0],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    out_channels=reversed_block_out_channels[0],
                )
            )

        reversed_attention_levels = list(reversed(attention_levels))
        reversed_num_res_blocks = list(reversed(num_res_blocks))
        block_out_ch = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            block_in_ch = block_out_ch
            block_out_ch = reversed_block_out_channels[i]
            is_final_block = i == len(num_channels) - 1

            for _ in range(reversed_num_res_blocks[i]):
                blocks.append(
                    ResBlock(
                        spatial_dims=spatial_dims,
                        in_channels=block_in_ch,
                        norm_num_groups=norm_num_groups,
                        norm_eps=norm_eps,
                        out_channels=block_out_ch,
                    )
                )
                block_in_ch = block_out_ch

                if reversed_attention_levels[i]:
                    blocks.append(
                        AttentionBlock(
                            spatial_dims=spatial_dims,
                            num_channels=block_in_ch,
                            norm_num_groups=norm_num_groups,
                            norm_eps=norm_eps,
                            use_flash_attention=use_flash_attention,
                        )
                    )

            if not is_final_block:
                blocks.append(Upsample(spatial_dims=spatial_dims, in_channels=block_in_ch))

        blocks.append(nn.GroupNorm(num_groups=norm_num_groups, num_channels=block_in_ch, eps=norm_eps, affine=True))
        blocks.append(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=block_in_ch,
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class Upsample(nn.Module):
    """
    Convolution-based upsampling layer.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D, 3D).
        in_channels: number of input channels to the layer.
    """

    def __init__(self, spatial_dims: int, in_channels: int) -> None:
        super().__init__()
        # Using transposed conv instead of F.interpolate
        self.conv = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=in_channels,
            strides=2,
            kernel_size=3,
            padding=1,
            conv_only=True,
            is_transposed=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        # dtype = x.dtype
        # if dtype == torch.bfloat16:
        #     x = x.to(torch.float32)
        #
        # x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        #
        # # If the input is bfloat16, we cast back to bfloat16
        # if dtype == torch.bfloat16:
        #     x = x.to(dtype)

        x = self.conv(x)
        return x


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)
