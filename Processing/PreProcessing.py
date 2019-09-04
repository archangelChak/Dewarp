from torchvision import transforms
from scipy.signal import convolve2d
from torchvision.transforms import Compose, Lambda
import numpy as np


def Smooth(mask_in):
    smoothed_mask = mask_in.copy()
    smoothed_mask[smoothed_mask < -0.1] = 1e4
    kernel = np.ones((5,5),np.float32)/25.
    smoothed_mask[:, :, 0] = convolve2d(smoothed_mask[:, :, 0].astype(np.float32),kernel, mode='same')
    smoothed_mask[:, :, 1] = convolve2d(smoothed_mask[:, :, 1].astype(np.float32),kernel, mode='same')
    smoothed_mask[smoothed_mask > 1] = -1
    return smoothed_mask


ToTensor_pair = Lambda(
    lambda image_mask_pair: (transforms.ToTensor()(image_mask_pair[0]), transforms.ToTensor()(image_mask_pair[1]))
)
Transpose_pair = Lambda(
    lambda image_mask_pair: ((image_mask_pair[0]), image_mask_pair[1])
)

