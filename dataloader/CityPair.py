import os
import os.path.join as join
import torch
import numpy as np
from skimage.io import imread, imsave
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
class CityPairsDatasets(Dataset):
    """Semantic Segmentation dataset."""

    def __init__(self, path_to_dataset, set_name, transform = None):
        """
        Args:
        path_to_dataset: folder where dataset is stored:
        Folder Template
        Dataset_Folder
        |___
        Train
            |_ Images
            |  ...jpg
            |  ...jpg
            |_ Mask
            |  ...jpg
            |  ...jpg
            |
        Test
            |_ Images
            |  ...jpg
            |  ...jpg
            |_ Mask
            |  ...jpg
            |  ...jpg
            |
            
       
        
        """
        self.transform = transform
        self.root      = path_to_dataset
        self.set_name  = set_name 
        self.imagename  = os.listdir(join(self.root, self.set_name))
        
    def __len__(self):
        return len(self.imagename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = join(self.root, self.set_name, self.imagename[idx])
        image = imread(img_name)

        if self.transform:
            
            image = self.transform(image)
            return image[:,0:image.shape[1]/2,:], image[:,image.shape[1]/2:,:]
        else:
        	image = np.asarray(image)
        	return image[:,0:image.shape[1]/2,:], image[:,image.shape[1]/2:,:]
