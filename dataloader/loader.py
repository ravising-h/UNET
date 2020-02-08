import os
import os.path.join as join
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
class SemanticSegmentation(Dataset):
    """Semantic Segmentation dataset."""

    def __init__(self, path_to_dataset, set_name, transform = None):
        """
        Args:
        path_to_dataset: folder where dataset is stored:
        Folder Template
        Dataset_Folder
        |___
        	|
            |_ Images
            |  ...jpg
            |  ...jpg
            |_ Mask
            |  ...jpg
            |  ...jpg
            |
            |_train.txt
            |_test.txt
        set_name = "train" or "test"
        
        """
        self.transform = transform
        self.root      = path_to_dataset
        self.set_name  = set_name 
        self.filename  = os.listdir(join(self.root, "Images"))
        
        try:
            with open(join(self.root,str(self.set_name) + ".txt","r")) as file:
                self.imagename = file.read().split()
        except:
            raise Exception("{}.txt does not exists in {}".format(self.set_name,self.root))

        if len(self.filename) != len(os.listdir(join(self.root , "Mask"))):
            raise Exception('X != Y : Len of Images and Mask is not same!')
        
    def __len__(self):
        return len(self.imagename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = join(self.root,"Images",self.imagename[idx])
        image = Image.open(img_name)
        msk_name = join(self.root,"Mask",self.imagename[idx])
        mask = Image.open(msk_name)

        if self.transform:
            image = self.transform(image)
            mask =  self.transform(mask)
        else:
        	image = np.asarray(image)
        	mask = np.asarray(mask)
        return image, mask

