#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append(os.getcwd())

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import datasets.custom_transforms as c_transforms


class FaceSynthetic(Dataset):
    def __init__(self, general_dict, cropsize=(640, 480), mode='train'):
        super(FaceSynthetic, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.general_dict = general_dict
        self.imgs = os.listdir(general_dict['img_path'])

        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4207, 0.3943, 0.3677), (0.2562, 0.2576, 0.2705))    #v2
            # transforms.Normalize((0.3677, 0.3943, 0.4207), (0.2705, 0.2576, 0.2562))  #v1

            ])
        self.trans_train_main = c_transforms.Compose([
            c_transforms.HorizontalFlip(),
            c_transforms.RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            c_transforms.RandomCrop(cropsize)
            ])
        self.trans_train_color = c_transforms.Compose([
            c_transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5)])

    def __getitem__(self, idx) -> dict:
        img_idx = self.imgs[idx]
        img = Image.open(os.path.join(self.general_dict['img_path'], img_idx))

        img = img.resize((512, 512), Image.BILINEAR)
        label_path = os.path.join(self.general_dict['mask_path'],
                                  img_idx[:-4]) + '_seg.png'
        label = Image.open(label_path).convert('P')

        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train_main(im_lb)
            og_img = im_lb['im'].copy()
            og_img = transforms.ToTensor()(og_img)
            img = self.trans_train_color(im_lb)
            img, label = im_lb['im'], im_lb['lb']

        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return {'img': img, 'label': label, 'og_img': og_img}

    def __len__(self):
        return len(self.imgs)
