import os
import torch
import numpy as np


def rect_subsample_mask(image_shape, ss_stride=10):
    D, H, W = image_shape
    mask = np.ones(image_shape)
    for i in range(D):
        offset = i % ss_stride
        zero_idx = (np.arange(H)-offset)%ss_stride != 0
        mask[i, zero_idx, :] = 0.
    return mask
    

def triangle_subsample_mask(image_shape, ss_stride=10):
    D, H, W = image_shape
    w_stride = int(W / (ss_stride / 2))

    not_divisible = W % w_stride != 0

    mask = np.zeros(image_shape)
    for d in range(D):
        offset_up = (ss_stride - d) % ( ss_stride / 2) * 2 - 1
        offset_down = (ss_stride - d) % ( ss_stride / 2) * 2 + ss_stride
        for s_id in range((ss_stride // 2)):
            offset_up += 1
            up_one_idx = (np.arange(H)-offset_up)%ss_stride == 0
            mask[d, up_one_idx, s_id * w_stride:(s_id+1) * w_stride] = 1

            offset_down -= 1
            down_one_idx = (np.arange(H)-offset_down)%ss_stride == 0
            mask[d, down_one_idx, s_id * w_stride:(s_id+1) * w_stride] = 1

        if not_divisible:
            mask[d, up_one_idx, (s_id+1) * w_stride:W] = 1
            mask[d, down_one_idx, (s_id+1) * w_stride:W] = 1

    return mask


def generate_mask(image_shape, ss_stride=10, mask_type='rectangle'):
    if mask_type =='rectangle':
        mask = rect_subsample_mask(image_shape, ss_stride)
    elif mask_type == 'triangle':
        mask = triangle_subsample_mask(image_shape, ss_stride)
    else:
        raise ValueError('Invalid mask type: {}'.format(mask_type))
    return mask
