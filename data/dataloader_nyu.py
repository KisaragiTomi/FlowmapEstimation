import random
import numpy as np
from PIL import Image
from imageio import imread, imwrite
import imageio
from scipy.ndimage import zoom
import scipy
import matplotlib.pyplot as plt
import cv2

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import data.utils as data_utils


# Modify the following
NYU_PATH = './datasets/nyu/'


class NyuLoader(object):
    def __init__(self, args, mode):
        """mode: {'train_big',  # training set used by GeoNet (CVPR18, 30907 images)
                  'train',      # official train set (795 images) 
                  'test'}       # official test set (654 images)
        """
        self.t_samples = NyuLoadPreprocess(args, mode)

        # train, train_big
        if 'train' in mode:
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False)


class NyuLoadPreprocess(Dataset):
    def __init__(self, args, mode):
        self.args = args
        # train, train_big, test, test_new
        with open("./data_split/nyu_%s.txt" % mode, 'r') as f:
            self.filenames = f.readlines()
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.dataset_path = NYU_PATH
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]

        # img path and norm path
        img_path = self.dataset_path + '/' + sample_path.split()[0]
        norm_path = self.dataset_path + '/' + sample_path.split()[1]
        height_path = self.dataset_path + '/' + sample_path.split()[2]
        scene_name = self.mode
        img_name = img_path.split('/')[-1].split('.png')[0]

        # read img / normal
        imga = imread(img_path).astype(np.float32)
        heighta = imread(height_path).astype(np.float32)
        #imga = Image.open(img_path).convert("RGBA").resize(size=(self.input_width, self.input_height),resample=Image.BILINEAR)
        norm_gta = Image.open(norm_path).convert("RGBA").resize(size=(self.input_width, self.input_height),
                                                            resample=Image.BILINEAR)

        # horizontal flip (default: True)
        DA_hflip = False
        if self.args.data_augmentation_hflip:
            #DA_hflip = random.random() > 0.5
            if DA_hflip:
                img = TF.hflip(img)
                norm_gt = TF.hflip(norm_gt)

        # to array
        scaling_factors = [new_size / old_size for new_size, old_size in zip((self.input_height, self.input_width), imga.shape)]
        img = zoom(imga, zoom=scaling_factors)
        height_gt = zoom(heighta, zoom=scaling_factors)
        img = img.reshape(self.input_height, self.input_width, 1)
        height_gt = height_gt.reshape(self.input_height, self.input_width, 1)

        #img.repeat(3, axis=-1)

        img = img / 65535.0
        height_gt = height_gt/ 65535.0
        ww = img[:, :, 0]
        #ww = np.repeat(ww, 3, axis=2)

        norm_gta = np.array(norm_gta).astype(np.uint8)
        norm_gt = norm_gta[:, :, :3]
        ww_gt = norm_gta[:, :, 3]
        #ww_gt = np.repeat(ww_gt, 3, axis=2)


        norm_valid_mask = img[:, :, 0] != 0
        norm_valid_mask = norm_valid_mask[:, :, np.newaxis]
        norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0

        if DA_hflip:
            norm_gt[:, :, 0] = -norm_gt[:, :, 0]

        # random crop (default: False)
        if self.args.data_augmentation_random_crop:
            img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask,
                                                                 height=416, width=544)

        # color augmentation (default: True)
        # if self.args.data_augmentation_color:
        #     if random.random() > 0.5:
        #         img = data_utils.color_augmentation(img, indoors=True)

        #Image.fromarray((img*255).squeeze().astype(np.uint8), mode='L').save('/home/kalou/GithubP/FlowmapEstimation/img.png')
        #Image.fromarray((norm_gt*255).squeeze().astype(np.uint8), mode='L').save('/home/kalou/GithubP/FlowmapEstimation/norm_gt.png')
        # to tensors
        img = self.normalize(torch.from_numpy(img).repeat(1,1,3).permute(2, 0, 1))          # (3, H, W)
        height_gt = torch.from_numpy(height_gt).permute(2, 0, 1)         # (1, H, W)

        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        ww = torch.from_numpy(ww)                            # (1, H, W)
        ww_gt = torch.from_numpy(ww_gt)                  # (1, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'height': height_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': scene_name,
                  'img_name': img_name,
                  'ww': ww,
                  'ww_gt': ww_gt}

        return sample
