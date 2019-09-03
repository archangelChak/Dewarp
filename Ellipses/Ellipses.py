from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Ellipses.Transform import RandomD4, RandomD4_for_pair
from torchvision.transforms import Compose, Lambda
import numpy as np
from PIL import Image, ImageDraw
import torch

class Ellipses(Dataset):
    def __init__(self, num_images, image_shape=(64, 64), min_ellipses_num=5, max_ellipses_num=10,
                 transform=None):
        super().__init__()
        self.image_shape = image_shape
        self.min_ellipses_num = min_ellipses_num
        self.max_ellipses_num = max_ellipses_num
        
        self._image_mask_pairs = []
        
        for i in range(num_images):
            self._image_mask_pairs.append(self._create_one_image())
        
        self.transform = transform
        
    def _get_random_color(self):
        return tuple(np.random.randint(0, 255, 3))
    
    def _get_random_ellipse(self):
        ellipse_size = [
            np.random.randint(self.image_shape[i] // 30, self.image_shape[i] // 3)
            for i in range(2)
        ]
        
        if ellipse_size[1] >= ellipse_size[0] * 1.5:
            ellipse_size[0] *= (ellipse_size[1] / ellipse_size[0]) * 0.9
        if ellipse_size[0] >= ellipse_size[1] * 1.5:
            ellipse_size[1] *= (ellipse_size[0] / ellipse_size[1]) * 0.9
        
        
        ellipse_first_coords = tuple([
            np.random.randint(0, self.image_shape[i] - ellipse_size[i])
            for i in range(2)
        ])
        
        ellipse_second_coords = tuple([
            ellipse_first_coords[i] + ellipse_size[i]
            for i in range(2)
        ])
                
        return ellipse_first_coords + ellipse_second_coords
        
    
    def _create_one_image(self):
        background_color = self._get_random_color()
        main_image = Image.new(mode="RGB", size=self.image_shape, color=background_color)
        mask_image = Image.new("1", self.image_shape, 1)

        draw_main = ImageDraw.Draw(main_image)
        draw_mask = ImageDraw.Draw(mask_image)
        
        num_ellipses = np.random.randint(self.min_ellipses_num, self.max_ellipses_num + 1)
        for i in range(num_ellipses):
            ellipse_coords = self._get_random_ellipse()
            ellipse_color = self._get_random_color()
            
            draw_main.ellipse(ellipse_coords, fill=ellipse_color, outline=ellipse_color)
            draw_mask.ellipse(ellipse_coords, fill=0, outline=0)
        
        type_of_noise = [
            0,
            np.random.randint(0, 100, self.image_shape)[:, :, np.newaxis],
            np.random.randint(0, 50, self.image_shape)[:, :, np.newaxis],
            np.random.randint(0, 50, (*self.image_shape, 3)),
            np.random.randint(0, 100, (*self.image_shape, 3)),
            np.abs(np.random.normal(0, 20, (*self.image_shape, 3))),
            np.abs(np.random.normal(0, 20, self.image_shape))[:, :, np.newaxis],
            np.abs(np.random.poisson(100, (*self.image_shape, 3))),
            np.abs(np.random.poisson(100, self.image_shape))[:, :, np.newaxis],
        ]
        noise = type_of_noise[np.random.randint(len(type_of_noise))]
        main_image = np.clip((np.array(main_image) + noise), 0, 255).astype('int64').astype('float64')
        mask_image = np.array(mask_image).astype('int64').astype('float64')
        
        return main_image, mask_image
    
    def __getitem__(self, ind):
        if self.transform is None:
            return self._image_mask_pairs[ind]
        else:
            return self.transform(self._image_mask_pairs[ind])
            #return self.transform(self._image_mask_pairs[ind])
        
    def __len__(self):
        return len(self._image_mask_pairs)
ToTensor_pair = Lambda(
    lambda image_mask_pair: (torch.Tensor(image_mask_pair[0]), torch.Tensor(image_mask_pair[1]))
)
Transpose_pair = Lambda(
    lambda image_mask_pair: ((image_mask_pair[0]).transpose(0, 2).transpose(1, 2), torch.Tensor(image_mask_pair[1]))
)
def CreateEllipsesDataset(num_images_train=100, num_images_test=30, resolution = (64,64)):
    e_train = Ellipses(num_images=100, image_shape = (64,64), transform=Compose([RandomD4_for_pair(), ToTensor_pair, Transpose_pair]))
    e_test = Ellipses(num_images=30, image_shape = (64,64), transform = Compose([ToTensor_pair, Transpose_pair]))
    ellipses_train_loader = DataLoader(e_train, batch_size=2, shuffle=False)
    ellipses_test_loader = DataLoader(e_test, batch_size=2, shuffle=False)
    return ellipses_train_loader, ellipses_test_loader