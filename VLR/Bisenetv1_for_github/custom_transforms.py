#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os, sys
from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random
import numpy as np
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import train.datasets.custom_transforms as c_transforms


class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h):
            return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im=im.crop(crop),
                lb=lb.crop(crop)
                    )


class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']

            # flip_lb = np.array(lb)

            # Since we are combining left and right features
            # flip_lb[lb == 2] = 3
            # flip_lb[lb == 3] = 2
            # flip_lb[lb == 4] = 5
            # flip_lb[lb == 5] = 4
            # flip_lb[lb == 7] = 8
            # flip_lb[lb == 8] = 7
            # flip_lb = Image.fromarray(flip_lb)
            return dict(im=im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb=lb.transpose(Image.FLIP_LEFT_RIGHT)
                        )
        
class VerticalFlip(object):
    def __init__(self, p=0.2, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            flip_lb = np.array(lb)

            flip_lb = Image.fromarray(flip_lb)
            return dict(im=im.transpose(Image.FLIP_TOP_BOTTOM),
                        lb=flip_lb.transpose(Image.FLIP_TOP_BOTTOM)
                        )
        

class RandomRotate(object):

  def __init__(self, degrees=(-45,45), *args, **kwargs):
      self.degrees = degrees

  def __call__(self, im_lb):

      im = im_lb['im']  
      lb = im_lb['lb']

      angle = random.uniform(self.degrees[0], self.degrees[1])

    #   im = Image.fromarray(im)
    #   lb = Image.fromarray(lb)
      
      im = im.rotate(angle, resample=Image.BILINEAR)
      lb = lb.rotate(angle, resample=Image.NEAREST)

    #   im = np.array(im)
    #   lb = np.array(lb)

      return {'im': im, 'lb': lb}

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im=im.resize((w, h), Image.BILINEAR),
                    lb=lb.resize((w, h), Image.NEAREST)
                    )


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None and brightness > 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if contrast is not None and contrast > 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if saturation is not None and saturation > 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im=im,
                    lb=lb
                    )


class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs


class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb

