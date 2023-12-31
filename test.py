import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from data.dataloader_custom import CustomLoader
from models.NNET import NNET
import utils.utils as utils
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test(model, test_loader, device, results_dir):
    alpha_max = 60
    kappa_max = 30

    with torch.no_grad():
        for index in range(len(test_loader.dataset)):
            print('process image' + str(index))
            data_dict = test_loader.dataset[index]
            img = data_dict['img'].to(device)
            img = img.unsqueeze(0)
            test = model(img)
            norm_out_list, _, _ = model(img)
            norm_out = norm_out_list[-1]

            pred_norm = norm_out[:, :3, :, :]
            pred_height = norm_out[:, [3], :, :]

            # to numpy arrays
            img = img.detach().cpu().permute(0, 2, 3, 1).numpy()                    # (B, H, W, 3)
            pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()        # (B, H, W, 3)
            pred_height = pred_height.cpu().permute(0, 2, 3, 1).numpy()
            pred_height = np.squeeze(pred_height)

            # save results
            img_name = data_dict['img_name']

            # 1. save input image
            img = utils.unnormalize(img[0, ...])
            img = Image.fromarray(img[:,:,0])
            target_path = '%s/%s_img.png' % (results_dir, img_name)
            #plt.imsave(target_path, img)
            img.save(target_path, "PNG", bits=16)
            #shutil.copy2()

            # 2. predicted normal
            pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
            pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
            pred_norm_rgb = pred_norm_rgb.astype(np.uint8)                  # (B, H, W, 3)
            pred_height *= 65535

            target_path = '%s/%s_pred_norm.png' % (results_dir, img_name)
            plt.imsave(target_path, pred_norm_rgb[0, :, :, :])

            # 3. predicted kappa (concentration parameter)
            # target_path = '%s/%s_pred_height.png' % (results_dir, img_name)
            # height = Image.fromarray(pred_height.astype(np.uint16))
            # height.save(target_path, bits=16)
#            plt.imsave(target_path, pred_kappa[0, :, :, 0], vmin=0.0, vmax=kappa_max, cmap='gray')

            # 4. predicted uncertainty
            #pred_alpha = utils.kappa_to_alpha(pred_kappa)
            target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
#            plt.imsave(target_path, pred_alpha[0, :, :, 0], vmin=0.0, vmax=alpha_max, cmap='jet')

            # 5. concatenated results
            # image_path_list = ['img', 'pred_norm', 'pred_alpha']
            # image_path_list = ['%s/%s_%s.png' % (results_dir, img_name, i) for i in image_path_list]
            # target_path = '%s/%s_concat.png' % (results_dir, img_name)
            # utils.concat_image(image_path_list, target_path)


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', default='GN', type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", default='nyu', type=str, help="{nyu, scannet}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--imgs_dir', default='./examples', type=str)

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    device = torch.device('cuda:0')

    # load checkpoint
    #checkpoint = './checkpoints/%s.pt' % args.pretrained
    checkpoint = './experiments/exp00_test/models/checkpoint_iter_0000034620.pt'
    print('loading checkpoint... {}'.format(checkpoint))
    model = NNET(args).to(device)
    model = utils.load_checkpoint(checkpoint, model)
    model.eval()
    print('loading checkpoint... / done')

    # test the model
    results_dir = args.imgs_dir + '/results'
    os.makedirs(results_dir, exist_ok=True)
    test_loader = CustomLoader(args, args.imgs_dir).data
    test(model, test_loader, device, results_dir)

