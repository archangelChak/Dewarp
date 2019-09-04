from skimage.transform import rotate
import numpy as np


class RandomD4(object):
    """
    Random D4 transformation
    
    param: object: image to transform
    """
    def __init__(self):
        self.possible_transformations = [
            lambda x: x,
            lambda x: rotate(x, 90, resize=False, preserve_range=True),
            lambda x: rotate(x, 180, resize=False, preserve_range=True),
            lambda x: rotate(x, 270, resize=False, preserve_range=True),
            lambda x: x[:, ::-1] - np.zeros_like(x),
            lambda x: x[::-1, :] - np.zeros_like(x),
            lambda x: rotate(x, 90, resize=False, preserve_range=True)[:, ::-1] - np.zeros_like(x),
            lambda x: rotate(x, 90, resize=False, preserve_range=True)[::-1, :] - np.zeros_like(x),
        ]

    def __call__(self, img):
        transformation_index = np.random.randint(len(self.possible_transformations))
        return self.possible_transformations[transformation_index](img)


class RandomD4_for_pair(RandomD4):
    """
    Random D4 transformation for pair of images
    
    param: RandomD4: transormation function
    """

    def __call__(self, img_mask_pair):
        transformation_index = np.random.randint(len(self.possible_transformations))
        transformed_image = self.possible_transformations[transformation_index](img_mask_pair[0])
        transformed_mask = self.possible_transformations[transformation_index](img_mask_pair[1])
        return transformed_image, transformed_mask
