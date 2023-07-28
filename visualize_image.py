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

import numpy as np
import torch
from monai.utils.type_conversion import convert_to_numpy
from monai.transforms import AsDiscrete
import torch.nn.functional as F

def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    # draw_img = image
    # if np.amin(draw_img) < 0:
    #     draw_img -= np.amin(draw_img)
    # if np.amax(draw_img) > 1:
    #     draw_img /= np.amax(draw_img)
    # draw_img = (255 * draw_img).astype(np.uint8)

    image = np.clip(image, -1., 1.)
    draw_img = ((image + 1.0) / 2.0 * 255).astype(np.uint8)

    # image = np.clip(image, 0., 1.)
    # draw_img = (image * 255).astype(np.uint8)
    return draw_img


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img

def normalize_label_to_uint8(colorize, label):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    post_label = AsDiscrete(to_onehot=5)
    label = post_label(label).permute(1, 0, 2, 3)
    label = F.conv2d(label, weight=colorize)
    label = torch.clip(label, -1, 1).squeeze().permute(1, 2, 0).cpu().numpy()

    draw_img = ((label + 1.0) / 2.0 * 255).astype(np.uint8)

    return draw_img
def visualize_one_slice_in_3d_label(colorize, image, axis: int = 2,):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """

    # draw image
    center = image.shape[2:][axis] // 2
    if axis == 0:
        draw_img = normalize_label_to_uint8(colorize,
                                            image[..., center, :, :])
    elif axis == 1:
        draw_img = normalize_label_to_uint8(colorize,
                                            image[..., :, center, :])
    elif axis == 2:
        draw_img = normalize_label_to_uint8(colorize,
                                            image[..., :, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    return draw_img

