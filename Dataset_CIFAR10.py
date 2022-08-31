import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import os
import cv2
from utils.util import *

img_w = 224
img_h = 224

data_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])


class CIFAR10Dataset(data.Dataset):
    def __init__(self, mode, dir):
        self.mode = mode
        self.list_img = []
        self.list_label = []
        self.dir_list = []
        self.data_size = 0
        self.transforms = data_transform
        self.dir = dir

        if self.mode == 'train':
            self.dir_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        elif self.mode == 'test':
            self.dir_list = ['test_batch']
        else:
            print('undefinde dataset!')

        for data_batch in self.dir_list:
            data_batch_path = os.path.join(self.dir, data_batch)
            batch_img = unpickle(data_batch_path)

            self.list_img.extend(batch_img[b'data'])
            self.list_label.extend(batch_img[b'labels'])

            self.data_size = self.data_size + batch_img[b'data'].shape[0]

    def __getitem__(self, item):

        data = self.list_img[item].reshape(3,32,32)
        data = data.transpose(1, 2, 0)

        data = cv2.resize(data, (img_w, img_h))
        label = self.list_label[item]

        return self.transforms(data), label

    def __len__(self):
        return self.data_size

