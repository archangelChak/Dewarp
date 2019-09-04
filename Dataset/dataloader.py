from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from PIL import Image
from os import listdir
from os.path import isfile, join
from Processing.PreProcessing import Smooth, ToTensor_pair, Transpose_pair
import numpy as np
from glob import glob

def get_names(path='/archive/docunet/warp_dataset/'):
    """
    Obtaining of names of images and masks for dynamic loader
    :return: tuple(imagenames,masksnames)
    """

    mask_names = glob('/archive/docunet/warp_dataset/*/masks/*').sort() 
    image_names = glob('/archive/docunet/warp_dataset/*/images/*').sort()  
    return (image_names, mask_names)


class Data(Dataset):
    """
    Creation of torch.utils.data.Dataset for input (loading + padding)

    :param num_images: int: number of images in train dataset
    :param images_in: tf.Tensor: list of names of images
    :param masks_in: list of names of masks
    :param transform: dict: transformation of input (to tensor)
    :return: torch.utils.data.Dataset
    """

    def __init__(self,num_images, images_in, masks_in, transform=None):
        super().__init__()
        self._image_mask_pairs = []
        set_ind = 0
        cont = 0
        flag = False
        for i in range(num_images):
            image_path = images_in[i]
            mask_path = masks_in[i]
            self._image_mask_pairs.append((image_path, mask_path))
        self.transform = transform
        
    def __getitem__(self, ind):
        image_path = self._image_mask_pairs[ind][0]
        mask_path = self._image_mask_pairs[ind][1]
        if self.transform is None:
            return (np.asarray(Image.open(image_path)), Smooth(np.array(np.load(mask_path), dtype=np.float32)))
        else:
            image=np.pad(array=np.asarray(Image.open(image_path)), pad_width=((0,36),(0,52)), mode='constant', constant_values=255)
            mask=np.pad(array=np.array(np.load(mask_path), dtype=np.float32), pad_width=((0,36),(0,52),(0,0)), mode='constant', constant_values=-1)
            return self.transform((image, mask))
        
    def __len__(self):
        return len(self._image_mask_pairs)


def get_loader(num_images_train=5000, num_images_test=1000):
    """
    Creation of torch.utils.data.DataLoader for input (division of Dataset by batches)

    :param num_images: int: number of images in train dataset
    :return: torch.utils.data.DataLoader
    """

    names = get_names()
    train = Data(num_images = num_images_train, images_in = names[0], masks_in = names[1], transform = Compose([ToTensor_pair, Transpose_pair]))
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    test = Data(num_images = num_images_train, images_in = names[0][num_images_train:], masks_in = names[1][num_images_train:], transform = Compose([ToTensor_pair, Transpose_pair]))
    test_loader = DataLoader(test, batch_size=2, shuffle=False)
    return train_loader
