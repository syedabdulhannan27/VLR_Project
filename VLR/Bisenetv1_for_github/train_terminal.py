import time
import os
# import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import train.datasets.dataset_factory as d_f
import train.models.model_factory as m_f
import train.losses.loss_factory as l_f
import train.tb_visualization.tensorboard_factory as tb_f
import train.saver.saver_factory as s_f
from tqdm import tqdm
import gc
from PIL import Image
import PIL.ImageEnhance as ImageEnhance


import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys
import json
from PIL import Image
from tqdm import tqdm
import random
import torchvision.transforms as transforms

import custom_transforms as c_transforms

# import configs.config_factory as c_f


# def visualize_dataloader(dataloader) -> None:
#     batch = next(iter(dataloader))

#     images, labels, og = batch['img'], batch['label'], batch['og_img']
#     grid_img = torchvision.utils.make_grid(images, nrow=8)

#     plt.figure(figsize=(15, 15))
#     plt.imshow(np.clip(np.transpose(grid_img, (1, 2, 0)), 0, 1))

#     grid_lb = torchvision.utils.make_grid(labels, nrow=8)

#     plt.figure(figsize=(15, 15))
#     plt.imshow((np.transpose(grid_lb, (1, 2, 0)))*15)

#     grid_og_img = torchvision.utils.make_grid(og, nrow=8)

#     plt.figure(figsize=(15, 15))
#     plt.imshow(np.clip(np.transpose(grid_og_img, (1, 2, 0)), 0, 1))


def training_loop(
    dataloaders: list(),
    network: nn.Module,
    criterion,
    optimizer,
    scheduler,
    training_dict: dict(),
    vis=None,
    saver=None
     ) -> nn.Module:

    # Training mode:
    network.train()

    for epoch in range(training_dict["num_epochs"]):

        # Start counting time:
        start = time.perf_counter()

        # Printing the epoch:
        print("Epoch {}/{}".format(epoch + 1, training_dict["num_epochs"]))
        print(len(dataloaders))
        for idx_batch, data_dict in enumerate(dataloaders):
            torch.cuda.empty_cache()
            # Set gradients to zero:
            optimizer.zero_grad()

            # Compute prediction:
            pred_dict = network(data_dict=data_dict)

            # Compute loss:
            loss = criterion.compute_loss(pred_dict=pred_dict)

            # Backpropagation:
            criterion.backward(loss)

            # Update the parameters:
            optimizer.step()

            # #Updating the tensorboard:
            if (vis is not None) and (idx_batch % training_dict['print_gap_batches'] == 0):
                if idx_batch == 0:
                    print(f'Batch {idx_batch}')
                    start_time = time.time()
                else:
                    end_time = time.time()
                    dur = end_time - start_time
                    print(f'Batch {idx_batch} | Duration:{dur:.2f}s')
                    start_time = time.time()
                # Calculating the iteration:
                iteration = epoch * len(dataloaders) + idx_batch

                # #Creating dict for the tensorboard:
                info_dict = {
                    'pred': pred_dict,
                    'loss': loss,
                    'data_dict': data_dict,
                    'epoch': epoch,
                    'lr': scheduler.get_last_lr()[0],
                    'iteration': iteration
                    }

                # Updating the tensorboard:
                vis.update(info_dict=info_dict)
                breakpoint
        
        ######################################################################
        # Saving the model:
        if saver is not None:
            saver.save(model=network,
                       optimizer=optimizer,
                       crrnt_loss=loss,
                       epoch=epoch
                       )
        ######################################################################

        # Update the scheduler:
        scheduler.step()

        # Stop counting time and printing it:
        time_epoch = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start))
        print(f'Time per epoch: {time_epoch}')
        print("-" * 10, flush=True)
        print()
    # Return the model:
    return network


# class CustomDataloader(torch.utils.data.DataLoader): # Inherit from torch.utils.data.DataLoader
#     def __init__(self, path_dict, batch_size, shuffle=True, augmentation=False, num_images_per_batch=1):

#         self.og_img_files = path_dict['og_img_files']
#         self.patch_folder = path_dict['patch_folder']
#         self.batch_size   = batch_size
#         self.shuffle      = shuffle
#         self.augmentation = augmentation
#         self.num_images_per_batch = num_images_per_batch

#         #  pre-processing
#         self.to_tensor = transforms.Compose([
#             transforms.ToTensor(),
#             # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ])
#         self.trans_train_main = c_transforms.Compose([
#             c_transforms.HorizontalFlip(),
#             c_transforms.RandomRotate(degrees=(-15,15)),
#             # c_transforms.RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
#             # c_transforms.RandomCrop(cropsize)
#             ])
#         self.trans_train_colors = transforms.Compose([
#             transforms.ColorJitter(
#                 brightness=0.1,
#                 contrast=0.1,
#                 saturation=0.1),
#             # transforms.RandomGrayscale(p=0.2),
#                 ])


#         # Each image gets split into sub images to form one training batch
#         self.num_batches =  len(os.listdir(self.patch_folder))//self.num_images_per_batch

#     def __len__(self):
#         # What output do you get when you print len(loader)? You get the number of batches
#         return self.num_batches

#     def __iter__(self):
#         data_dict = {}
#         batch_idx = 0

#         patch_folders = os.listdir(self.patch_folder)
        
#         if self.shuffle:
#             random.shuffle(patch_folders)

#         # patch_names = ([f for f in os.listdir(patch_folders)])
#         patch_paths = [os.path.join(self.patch_folder, file_name) for file_name in patch_folders]
        
#         while batch_idx < self.num_batches:
#             for i in range(self.num_images_per_batch):
#                 img_patches = sorted([os.path.join(patch_paths[batch_idx], f) for f in os.listdir(patch_paths[batch_idx]) if f.endswith('.jpg')])
#                 mask_patches = sorted([os.path.join(patch_paths[batch_idx], f) for f in os.listdir(patch_paths[batch_idx]) if f.endswith('.png')])
#                 data_dict_path = [os.path.join(patch_paths[batch_idx], f) for f in os.listdir(patch_paths[batch_idx]) if f.endswith('.json')]
                
#             with open(data_dict_path[0]) as json_file:
#                 data_dict = json.load(json_file)
#             img_batch = []
#             mask_batch = []
#             # img_and_mask_batch = []
#             for i in range(len(img_patches)):
#                 img = mpimg.imread(img_patches[i])
#                 mask = mpimg.imread(mask_patches[i])
#                 # print(img_patches[i])

            
#                 if not self.augmentation:
#                     augmented_img = self.to_tensor(img)
#                     augmented_mask= self.to_tensor(mask)
#                 else:
#                     img = Image.fromarray(img)
#                     mask = Image.fromarray(mask)
#                     im_lb = dict(im=img, lb=mask)
#                     im_lb = self.trans_train_main(im_lb)
#                     im_lb['im'] = self.trans_train_colors(im_lb['im'])

#                     augmented_img, augmented_mask = im_lb['im'], im_lb['lb']
                    
#                     augmented_img = self.to_tensor(np.array(augmented_img))
#                     augmented_mask= self.to_tensor(np.array(augmented_mask))
            
#             # APPLY AUGMENTATIONS FOLLOWED BY YIELDING
#                 # img_and_mask_batch.append({'img': augmented_img, 'label': augmented_mask, 'data_dict': data_dict})
#                 img_batch.append(augmented_img)
#                 mask_batch.append(augmented_mask)

#             img_batch = torch.stack(img_batch, dim=0)
#             mask_batch = torch.stack(mask_batch, dim=0)

#             batch_idx += self.num_images_per_batch
#             yield {'img': img_batch, 'label': mask_batch, 'data_dict': data_dict}

class CustomDataloader(torch.utils.data.DataLoader): # Inherit from torch.utils.data.DataLoader
    def __init__(self, path_dict, batch_size, shuffle=True, augmentation=False, num_images_per_batch=1):

        self.og_img_files = path_dict['og_img_files']
        self.patch_folder = path_dict['patch_folder']
        self.batch_size   = batch_size
        self.shuffle      = shuffle
        self.augmentation = augmentation
        self.num_images_per_batch = num_images_per_batch
    
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        # self.trans_train_main = c_transforms.Compose([
        #     c_transforms.HorizontalFlip(),
        #     c_transforms.RandomRotate(degrees=(-15,15)),
        #     # c_transforms.RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
        #     # c_transforms.RandomCrop(cropsize)
        #     ])
        self.trans_train_colors = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1),
            # transforms.RandomGrayscale(p=0.2),
                ])


        # Each image gets split into sub images to form one training batch
        self.num_batches =  len(os.listdir(self.patch_folder))//self.num_images_per_batch

    def __len__(self):
        # What output do you get when you print len(loader)? You get the number of batches
        return self.num_batches

    def __iter__(self):
        data_dict = {}
        batch_idx = 0

        patch_folders = os.listdir(self.patch_folder)
        
        if self.shuffle:
            random.shuffle(patch_folders)

        # patch_names = ([f for f in os.listdir(patch_folders)])
        patch_paths = [os.path.join(self.patch_folder, file_name) for file_name in patch_folders]
        
        while batch_idx < self.num_batches:

            img_patches_combined = []
            mask_patches_combined = []
            data_dict_list = []
            for i in range(self.num_images_per_batch):
                img_patches = sorted([os.path.join(patch_paths[batch_idx+i], f) for f in os.listdir(patch_paths[batch_idx+i]) if f.endswith('.jpg')])
                mask_patches = sorted([os.path.join(patch_paths[batch_idx+i], f) for f in os.listdir(patch_paths[batch_idx+i]) if f.endswith('.png')])
                data_dict_path = [os.path.join(patch_paths[batch_idx+i], f) for f in os.listdir(patch_paths[batch_idx+i]) if f.endswith('.json')]
                
                with open(data_dict_path[0]) as json_file:
                    data_dict = json.load(json_file)
                data_dict_list.append(data_dict)
                img_patches_combined.extend(img_patches)
                mask_patches_combined.extend(mask_patches)
                # print(f'Combined img patches = {len(img_patches_combined)}')

            
            img_batch = []
            mask_batch = []
            # img_and_mask_batch = []
            for i in range(len(img_patches_combined)):
                img = mpimg.imread(img_patches_combined[i])
                mask = mpimg.imread(mask_patches_combined[i])
                # print(img_patches[i])

            
                if not self.augmentation:
                    augmented_img = self.to_tensor(img)
                    augmented_mask= self.to_tensor(mask)
                else:
                    img = Image.fromarray(img)
                    mask = Image.fromarray(mask)
                    im_lb = dict(im=img, lb=mask)
                    # im_lb = self.trans_train_main(im_lb)
                    im_lb['im'] = self.trans_train_colors(im_lb['im'])

                    augmented_img, augmented_mask = im_lb['im'], im_lb['lb']
                    
                    augmented_img = self.to_tensor(np.array(augmented_img))
                    augmented_mask= self.to_tensor(np.array(augmented_mask))
            
            # APPLY AUGMENTATIONS FOLLOWED BY YIELDING
                # img_and_mask_batch.append({'img': augmented_img, 'label': augmented_mask, 'data_dict': data_dict})
                img_batch.append(augmented_img)
                mask_batch.append(augmented_mask)

            img_batch = torch.stack(img_batch, dim=0)
            mask_batch = torch.stack(mask_batch, dim=0)

            batch_idx += 1
            yield {'img': img_batch, 'label': mask_batch, 'data_dict': data_dict_list}
            # yield img_and_mask_batch


def train_run(general_dict):

    # Getting dataset:
    dataset, info_dataset_dict = d_f.dataset_factory(
        type_dataset=general_dict['type_dataset'],
        conf=general_dict
        )

    # Creating dataloader:
    # dataloader = DataLoader(
    #     dataset, 
    #     batch_size=general_dict['dataset_dict']["batch_size"], 
    #     shuffle=general_dict['dataset_dict']["shuffle"], 
    #     num_workers=general_dict['dataset_dict']["num_workers"],
    #     pin_memory=general_dict['dataset_dict']["pin_memory"]
    #     )

    path_dict = {
    "og_img_files" : '/home/ishan/datasets/celeba/images',
    "patch_folder" : '/home/ishan/datasets/celeba/patch_folder',
    }

    dataloader = CustomDataloader(path_dict=path_dict, batch_size=16, shuffle=False, augmentation=True, num_images_per_batch=50)
    print('Number of images to train:', len(dataloader))

    # Visualize dataloader images in a grid:
    # visualize_dataloader(dataloader)

    # Model:
    network_object = m_f.model_factory(
        type_model=general_dict['type_model'],
        conf=general_dict
        )
    network_object.to(general_dict['device'])

    # Criterion:
    loss_object = l_f.loss_factory(
        type_loss=general_dict['type_loss'],
        conf=general_dict
        )

    # Optimizer:
    optimizer = torch.optim.Adam(
        params=network_object.parameters(),
        lr=general_dict['training_dict']["learning_rate"]
        )

    # Scheduler:
    opt_lr_scheduler = lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=general_dict['cfg']["gamma_scheduler"]
        )

    # Tensorboard:
    vis_object = tb_f.tensorboard_factory(
        type_vis=general_dict['type_vis'],
        conf=general_dict
        )

    # Saver for the model:
    saver_object = s_f.save_factory(
        type_saver=general_dict['type_saver'],
        conf=[general_dict, info_dataset_dict]
        )

    # Start counting training time:
    start_time = time.perf_counter()
    print("Start training!")

    if general_dict['load_from_snapshot']:
        checkpoint = torch.load(general_dict['resume_path']) 
        network_object.load_state_dict(checkpoint)
        optimizer = torch.optim.Adam(
            params=network_object.parameters(),
            lr=1e-4,
            weight_decay=1e-6
            )
        opt_lr_scheduler = lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=0.975
            )

    torch.cuda.empty_cache()
    gc.collect()

    # Training loop:
    model = training_loop(
        dataloaders=dataloader,
        network=network_object,
        criterion=loss_object,
        optimizer=optimizer,
        scheduler=opt_lr_scheduler,
        vis=vis_object,
        training_dict=general_dict['training_dict'],
        saver=saver_object
        )

    respth = general_dict["model_path"]
    save_pth = os.path.join(respth, f'{general_dict["tb_name"]}.pth')
    state = model.state_dict()
    torch.save(state, save_pth)

    # Finish counting training time:
    total_time = time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))
    print(f'Total training time: {total_time}')
    print("Done!")