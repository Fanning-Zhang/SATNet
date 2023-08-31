"""
Load images/labels/disparity_images from an annotation file.
This code is only for LIVE 3D dataset.

The list file is like:
    left_img.bmp left_dis_img.mat right_img.bmp right_dis_img.mat label
"""


# make print_function compatible with python2 and python3
from __future__ import print_function

import os
import sys
import numpy as np
from scipy.io import loadmat

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class PreProcessDataset(data.Dataset):
    """
    Since Train and test have the same data pre-processing,
    one class PreProcessDataset is enough,
    it is fine to call this class for both TrainDataset and TestDataset.
    """
    def __init__(self, root_src, list_file, train, transform_l, transform_r):
        """
        Args:
          root_src: (str) directory to .txt.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        """
        self.root_src = root_src
        self.train = train
        self.transform_l = transform_l
        self.transform_r = transform_r

        # read image path of left and right, and label from txt file.
        self.fnames_l = []
        self.fnames_r = []

        self.labels = []

        with open(os.path.join('.', self.root_src, list_file), encoding='UTF-8-sig') as f:
            lines = f.readlines()   # read all lines
            self.num_samples = len(lines)

        for line in lines:

            # strip removes spaces at the beginning and end of the string
            # split the string based on four ' '(spaces)
            img_path_l, dis_path_l, img_path_r, dis_path_r, label = line.strip().split(' ', 4)

            self.fnames_l.append(img_path_l)
            self.fnames_r.append(img_path_r)

            label = float(label)
            self.labels.append(label)

    def __getitem__(self, idx):
        """
        Load image.
        Args:
            idx: (int) image index.
        Returns:
            img: (tensor) image tensor.
        """
        # Load image
        fname_l = self.fnames_l[idx]
        fname_r = self.fnames_r[idx]

        # Load labels
        train_label = self.labels[idx]

        # open image, mode=RGB, (tuple)
        img_l = Image.open(fname_l)
        img_r = Image.open(fname_r)

        # (if HSV, etc.) convert to RGB
        if img_l.mode != 'RGB':
            img_l = img_l.convert('RGB')
        if img_r.mode != 'RGB':
            img_r = img_r.convert('RGB')

        # transform image
        img_l = self.transform_l(img_l)
        img_r = self.transform_r(img_r)

        return img_l, img_r, train_label

    def collate_fn(self, batch):
        """
        Integrate the date into a batch.
        Args:
            batch: (list) of images
        Returns:
            image batch (Tensor), target_label (Tensor)
        """
        img_l = [x[0] for x in batch]
        img_r = [x[1] for x in batch]

        img_label = [x[2] for x in batch]

        c, h, w = img_l[0].size()
        num_imgs = len(img_l)

        inputs_l = torch.zeros(num_imgs, 3, h, w)
        inputs_r = torch.zeros(num_imgs, 3, h, w)

        target_label = torch.zeros(num_imgs)

        for i in range(num_imgs):

            inputs_l[i] = img_l[i]
            inputs_r[i] = img_r[i]

            target_label[i] = img_label[i]

        return inputs_l, inputs_r, target_label

    def __len__(self):
        return self.num_samples
