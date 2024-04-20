#!/usr/bin/python
# -*- encoding: utf-8 -*-

# from logger import setup_logger
from train.models.bisenetv1 import BiSeNet
import torch

import os
import os.path as osp
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

device_num = 0

# args
parse = argparse.ArgumentParser()
parse.add_argument('--dspth', dest='dspth', type=str)
parse.add_argument('--respth', dest='respth', type=str)
parse.add_argument('--cp_path', dest='cp_path', type=str)
args = parse.parse_args()


def vis_parsing_maps(
        im: Image,
        parsing_anno: np.ndarray,
        stride: int,
        save_path: str,
        save_im: bool = False
        ) -> None:
    """
    Visualizes parsing annotations on an image using color mapping and optional saving.

    Args:
        im : The input image.
        parsing_anno : Parsing annotation array containing class labels.
        stride : Scaling factor for resizing the parsing annotation.
        save_path : Path to save the visualization image.
        save_im : Whether to save the visualization image. Default is False.

    Returns:
        None

    This function visualizes parsing annotations on an input image by assigning colors to different
    parts based on provided class labels. The visualization can be optionally saved to a file.
    The parsed image can be displayed or saved depending on the 'save_im' parameter.
    """
    # Colors for all 20 parts
    part_colors = [
        [255, 125, 125], [215, 235, 255], [50, 65, 225],
        [0, 140, 255], [0, 252, 124],
        [167, 108, 188], [66, 71, 147], [170, 255, 0],
        [19, 69, 139], [0, 255, 170],
        [0, 0, 255], [85, 219, 236], [170, 0, 255],
        [0, 85, 255], [0, 170, 255],
        [255, 255, 0], [255, 255, 85], [255, 255, 170],
        [255, 0, 255], [255, 85, 255], [255, 170, 255],
        [0, 255, 255], [85, 255, 255], [170, 255, 255]
    ]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride,
        interpolation=cv2.INTER_NEAREST
    )
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0],
         vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(
        cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR),
        0.5,
        vis_parsing_anno_color,
        0.5,
        0
    )
    print('Parsing anno shape:', vis_parsing_anno.shape)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] + '.png', vis_parsing_anno*25)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def evaluate(respth, dspth, cp):

    print(f'respth:{respth},\ndspth:{dspth},\ncp:{cp}\n')
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes, mode='infer')
    net.cuda(device_num)
    model_save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(model_save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4207, 0.3943, 0.3677),
            (0.2562, 0.2576, 0.2705)
            )])

    with torch.no_grad():
        for image_path in os.listdir(dspth):
            if image_path.endswith('.png') or image_path.endswith('.jpg'):
                img = Image.open(osp.join(dspth, image_path))
                image = img  # img.resize((512, 512), Image.BILINEAR)

                min_dim = 512  # Minimum dimension we want to scale to
                width, height = image.size

                # Calculate the scale factor
                scale = min_dim / min(width, height)

                # Get the new width and height
                new_width, new_height = int(width * scale), int(height * scale)

                # Resize image maintaining aspect ratio
                image = image.resize((new_width, new_height))

                img = to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda(device_num)

                # out = net({'img': img})['out']
                # parsing = out.squeeze(0).cpu().numpy().argmax(0)

                # out_inter = out.argmax(1)
                out = net({'img': img})['out_single_channel']
                parsing = out.squeeze(0).cpu().numpy()


                print(np.unique(parsing))
                print(osp.join(respth, image_path))

                vis_parsing_maps(
                    image,
                    parsing,
                    stride=1,
                    save_im=True,
                    save_path=osp.join(respth, image_path)
                    )


if __name__ == "__main__":
    evaluate(
        respth=args.respth,
        dspth=args.dspth,
        cp=args.cp_path
        )
