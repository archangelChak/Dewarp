from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from PIL import Image
from os import listdir
from os.path import isfile, join
from Processing.PreProcessing import Smooth, ToTensor_pair, Transpose_pair
import numpy as np

def getNames():
    """
    Obtaining of names of images and masks for dynamic loader
    :return: tuple(imagenames,masksnames)
    """
    masknames=[]
    imagenames=[]
    for i in range(17):
        masknames.append([f for f in listdir('/archive/docunet/warp_dataset/{}/masks'.format(i)) if isfile(join('/archive/docunet/warp_dataset/{}/masks'.format(i), f))])
        imagenames.append([f for f in listdir('/archive/docunet/warp_dataset/{}/images'.format(i)) if isfile(join('/archive/docunet/warp_dataset/{}/images'.format(i), f))])
        masknames[-1].sort()
        imagenames[-1].sort()    
    return (imagenames, masknames)

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
        while True:
            path_images = '/archive/docunet/warp_dataset/{}/images/'.format(set_ind)
            path_masks = '/archive/docunet/warp_dataset/{}/masks/'.format(set_ind)
            for i in range(len(images_in[set_ind])):
                image_path = path_images + images_in[set_ind][i]
                mask_path = path_masks + masks_in[set_ind][i]
                self._image_mask_pairs.append((image_path, mask_path))
                cont+=1
                if (cont>=num_images):
                    flag = True
            if (flag): break
        self.transform = transform
        
    def __getitem__(self, ind):
        if self.transform is None:
            image_path = self._image_mask_pairs[ind][0]
            mask_path = self._image_mask_pairs[ind][1]
            return (np.asarray(Image.open(image_path)), Smooth(np.array(np.load(mask_path), dtype=np.float32)))
        else:
            image_path = self._image_mask_pairs[ind][0]
            mask_path = self._image_mask_pairs[ind][1]
            image=np.pad(array=np.asarray(Image.open(image_path)), pad_width=((0,36),(0,52)), mode='constant', constant_values=255)
            mask=np.pad(array=np.array(np.load(mask_path), dtype=np.float32), pad_width=((0,36),(0,52),(0,0)), mode='constant', constant_values=-1)
            return self.transform((image, mask))
        
    def __len__(self):
        return len(self._image_mask_pairs)

def GetLoader(num_images=5000):
    """
    Creation of torch.utils.data.DataLoader for input (division of Dataset by batches)

    :param num_images: int: number of images in train dataset
    :return: torch.utils.data.DataLoader
    """
    names = getNames()
    train = Data(num_images = num_images, images_in = names[0], masks_in = names[1], transform = Compose([ToTensor_pair, Transpose_pair]))
    train_loader = DataLoader(train, batch_size=2, shuffle=False)
    return train_loader