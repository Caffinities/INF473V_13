from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from hydra.utils import instantiate
import torch
#from RandAugment import RandAugment

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch

class unlabelledDataset(Dataset):
    def __init__(self, test_dataset_path, test_transform):
        self.test_dataset_path = test_dataset_path
        self.test_transform = test_transform
        images_list = os.listdir(self.test_dataset_path)
        # filter out non-image files
        self.images_list = [image for image in images_list if image.endswith(".jpg")]
        # set a non exist label
        self.labels = [48] * len(self.images_list)
        self.labels = torch.tensor(self.labels).type(torch.LongTensor)

    def __getitem__(self, idx):
        image_name = self.images_list[idx]
        label = self.labels[idx]
        image_path = os.path.join(self.test_dataset_path, image_name)
        image = Image.open(image_path)
        image = self.test_transform(image)
        return image, label, idx

    def __len__(self):
        return len(self.images_list)
    
    def set_label(self, newlabel, idx):
         self.labels[idx] = newlabel

class combinedDataset(Dataset):
    def __init__(self, train_dataset, unlabelleddataset, unlabelled_total):
            self.train_dataset = train_dataset
            self.unlabelleddataset = unlabelleddataset
            self.indexs = torch.tensor([i for i in range(len(self.unlabelleddataset))]).type(torch.LongTensor)
            self.labels = torch.tensor([-1 for i in range(len(self.unlabelleddataset))]).type(torch.LongTensor)
            permutation = torch.randperm(len(self.unlabelleddataset))
            self.indexs = permutation[0:unlabelled_total]
            #print(self.images.shape)
    def __getitem__(self, idx):
        if idx<len(self.train_dataset):
             image, label = self.train_dataset.__getitem__(idx)
             label = torch.tensor(label)
        else :
            trueindex = self.indexs[idx-len(self.train_dataset)]
            image, label, idxofdataset = self.unlabelleddataset.__getitem__(trueindex)
            label = self.labels[idx-len(self.train_dataset)]
        return image, label
    def __len__(self):
         return torch.tensor(len(self.indexs)+len(self.train_dataset))
    def adddata(self, idx, label):
        self.indexs.append(idx)
        self.labels = torch.cat((self.labels, label.unsqueeze(0)), dim=0).type(torch.LongTensor)
    def resetlabel(self, newlabel, idx):
         self.labels[idx] = newlabel

    
class DataModule:
    def __init__(
        self,
        train_dataset_path,
        unlabelled_dataset_path,
        train_transform,
        val_transform,
        aug_transform,
        aug_transform_strong,
        batch_size,
        num_workers,
    ):
        self.dataset = ImageFolder(train_dataset_path, transform=train_transform)
        
        '''self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset,
            [
                int(0.8 * len(self.dataset)),
                len(self.dataset) - int(0.8 * len(self.dataset)),
            ],
            generator=torch.Generator().manual_seed(3407),
        )'''
        self.train_dataset = self.dataset
        self.val_dataset = self.dataset
        self.aug_transform = aug_transform
        self.val_dataset.transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.strong_transform = aug_transform_strong
        #self.strong_transform.transforms.insert(0, RandAugment(4, 5))
        self.unlabelled_dataset_path = unlabelled_dataset_path
        self.unlabelled_transform = train_transform
        self.unlabelled_dataset = unlabelledDataset(self.unlabelled_dataset_path, self.unlabelled_transform)

    def data_augment(self, pic):
            pic_augment = self.aug_transform(pic)
            return pic_augment

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    def unlabelled_dataloader(self):
         return DataLoader(self.unlabelled_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        )