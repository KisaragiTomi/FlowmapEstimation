import argparse
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import math

import utils.utils as utils
from utils.losses import compute_loss

import matplotlib.pyplot as plt

def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


class ScaleInvariantError(nn.Module):
    def __init__(self, lamada=0.5):
        super(ScaleInvariantError, self).__init__()
        self.lamada = lamada
        return

    def forward(self, y_true, y_pred):
        first_log = torch.log(torch.clamp(y_pred, 0.0001))
        second_log = torch.log(torch.clamp(y_true, 0.0001))
        d = first_log - second_log
        loss = torch.mean(d * d) - self.lamada * torch.mean(d) * torch.mean(d)
        return loss


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.out(x)
        return torch.sigmoid(x)
def set_requires_grad(net, requires_grad=False):
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

def train(model, args, device):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    should_write = ((not args.distributed) or args.rank == 0)
    if should_write:
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # dataloader
    if args.dataset_name == 'nyu':
        from data.dataloader_nyu import NyuLoader
        train_loader = NyuLoader(args, 'train').data
        test_loader = NyuLoader(args, 'test').data
    else:
        raise Exception('invalid dataset name')

    # define losses
    loss_fn = compute_loss(args)

    discriminator = Discriminator()
    discriminator.apply(weights_init)
    discriminator = discriminator.cuda()
    discriminator = nn.DataParallel(discriminator)

    criterion_L1 = MaskedL1Loss()
    criterion_berHu = berHuLoss()
    criterion_MSE = MaskedMSELoss()
    criterion_SI = ScaleInvariantError()
    criterion_GAN = nn.BCELoss()
    criterion = nn.L1Loss()

    # optimizer
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": args.lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": args.lr}]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.lr)
    optimizer_D = optim.AdamW(discriminator.parameters(), lr=4 * 0.001)
    valid_T = torch.ones(args.batch_size, 1).cuda()
    zeros_T = torch.zeros(args.batch_size, 1).cuda()

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=args.lr,
                                              epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader),
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    scheduler_D = optim.lr_scheduler.OneCycleLR(optimizer=optimizer_D,
                                              max_lr=args.lr,
                                              epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader),
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)

    # cudnn setting
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # start training
    total_iter = 0
    model.train()

    for epoch in range(args.n_epochs):
        if args.rank == 0:
            t_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train")
        else:
            t_loader = train_loader
        test = 0
        for data_dict in t_loader:
            test += 1
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            total_iter += args.batch_size_orig

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)


            # forward pass
            if args.use_baseline:
                norm_out = model(img)
                loss = loss_fn(norm_out, gt_norm, gt_norm_mask)
                norm_out_list = [norm_out]
            else:
                norm_out_list, pred_list, coord_list = model(img, gt_norm_mask=gt_norm_mask, mode='train')

                if test == 50: #test
                    norm_out = norm_out_list[-1]
                    pred_norm = norm_out[:, :3, :, :]
                    # pred_norm
                    fixed_normal = gt_norm * 2 - 1
                    fixed_pred_norm = pred_norm * 2 - 1
                    fixed_normal = F.normalize(fixed_normal, p=2, dim=1)
                    fixed_pred_norm = F.normalize(fixed_pred_norm, p=2, dim=1)
                    dot = torch.cosine_similarity(fixed_pred_norm, fixed_normal, dim=1)
                    dot = (gt_norm[:, 0, :, :].detach() < 0.5).float()
                    dot = torch.cat([dot.unsqueeze(1)] * 3, dim=1)

                    # dot = torch.acos(dot)
                    #dot = (gt_norm[:, 0, :, :])
                    #dottest = dot[0, :, :]
                    v_gt_norm = F.normalize(gt_norm, p = 2, dim= 1)
                    v_gt_norm = v_gt_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
                    v_pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
                    v_dot = dot.detach().cpu().permute(0, 2, 3, 1).numpy()
                    v_dot *= 255
                    pred_norm_rgb = ((v_pred_norm + 1) * 0.5) * 255
                    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
                    pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
                    gt_norm_rgb = ((v_gt_norm + 1) * 0.5) * 255
                    gt_norm_rgb = np.clip(gt_norm_rgb, a_min=0, a_max=255)
                    gt_norm_rgb = gt_norm_rgb.astype(np.uint8)  # (B, H, W, 3)
                    v_dot = v_dot.astype(np.uint8)
                    # dot_norm_rgb = ((dot + 1) * 0.5) * 255
                    # dot_norm_rgb = np.clip(dot_norm_rgb, a_min=0, a_max=255)
                    # dot_norm_rgb = dot_norm_rgb.astype(np.uint8)  # (B, H, W, 3)

                    pred_target_path = './examples/results/pred_test.png'
                    gt_target_path = './examples/results/gt_test.png'
                    dot_target_path = './examples/results/dot_test.png'
                    plt.imsave(pred_target_path, pred_norm_rgb[0, :, :, :])
                    plt.imsave(gt_target_path, gt_norm_rgb[0, :, :, :])
                    #pred_alpha = utils.kappa_to_alpha(dot)
                    #target_path = '%s/%s_pred_alpha.png' % (results_dir, img_name)
                    #a = dot[0, :, :].cpu().detach().numpy()
                    #plt.imsave(dot_target_path, gt_norm_rgb[0, 0, :, :].cpu().detach().numpy())
                    gt_norm_rgb[0, :, :, 1] = gt_norm_rgb[0, :, :, 0]
                    gt_norm_rgb[0, :, :, 2] = gt_norm_rgb[0, :, :, 0]
                    plt.imsave(dot_target_path, v_dot[0, :, :, :])
                    test = 0

                loss = loss_fn(pred_list, coord_list, gt_norm, gt_norm_mask)


            loss_ = float(loss.data.cpu().numpy())
            if args.rank == 0:
                t_loader.set_description(f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train. Loss: {'%.5f' % loss_}")
                t_loader.refresh()

            # back-propagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            # lr scheduler
            scheduler.step()

            #patchGAN
            # loss_L1 = criterion_L1(pred, target)
            # loss_MSE = criterion_MSE(pred, target)
            # loss_berHu = criterion_berHu(pred, target)
            # loss_SI = criterion_SI(pred, target)
            #
            # set_requires_grad(discriminator, False)
            #
            # loss_adv = 0
            #
            # for a in range(12):
            #     for b in range(16):
            #         row = 19 * a
            #         col = 19 * b
            #         patch_fake = pred[:, :, row:row + 19, col:col + 19]
            #         pred_fake = discriminator(patch_fake)
            #         loss_adv += criterion_GAN(pred_fake, valid_T)
            #
            # loss_gen = loss_SI + 0.5 * loss_adv
            # loss_gen.backward()


            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad()
            loss_D = 0
            norm_out = norm_out_list[-1]
            pred = norm_out[:, :3, :, :]

            for a in range(12):
                for b in range(16):
                    row = 39 * a
                    col = 39 * b
                    pred_norm = pred[:, 0:3, :, :]
                    patch_fake = pred_norm[:, :, row:row + 39, col:col + 39]
                    patch_real = gt_norm[:, :, row:row + 39, col:col + 39]
                    pred_fake = discriminator(patch_fake.detach())
                    pred_real = discriminator(patch_real)
                    loss_D_fake = criterion_GAN(pred_fake, zeros_T)
                    loss_D_real = criterion_GAN(pred_real, valid_T)
                    loss_D += 0.5 * (loss_D_fake + loss_D_real)

            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(optimizer_D)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_D.step(optimizer_D)
            scaler_D.update()

            scheduler_D.step()



            # visualize
            if should_write and ((total_iter % args.visualize_every) < args.batch_size_orig):
                utils.visualize(args, img, gt_norm, gt_norm_mask, norm_out_list, total_iter)

            # save model
            if should_write and ((total_iter % args.validate_every) < args.batch_size_orig):
                model.eval()
                target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
                torch.save({"model": model.state_dict(),
                            "iter": total_iter}, target_path)
                print('model saved / path: {}'.format(target_path))
                validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)
                model.train()

                # empty cache
                torch.cuda.empty_cache()

    if should_write:
        model.eval()
        target_path = args.exp_model_dir + '/checkpoint_iter_%010d.pt' % total_iter
        torch.save({"model": model.state_dict(),
                    "iter": total_iter}, target_path)
        print('model saved / path: {}'.format(target_path))
        validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)

        # empty cache
        torch.cuda.empty_cache()

    return model


__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def _unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out

    
def validate(model, args, test_loader, device, total_iter, where_to_write, vis_dir=None):
    with torch.no_grad():
        total_normal_errors = None
        for data_dict in tqdm(test_loader, desc="Loop: Validation"):

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)

            # forward pass
            if args.use_baseline:
                norm_out = model(img)
            else:
                norm_out_list, _, _ = model(img, gt_norm_mask=gt_norm_mask, mode='test')
                norm_out = norm_out_list[-1]

            # upsample if necessary
            if norm_out.size(2) != gt_norm.size(2):
                norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)

            pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)

            prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            E = torch.acos(prediction_error) * 180.0 / np.pi

            mask = gt_norm_mask[:, 0, :, :]
            if total_normal_errors is None:
                total_normal_errors = E[mask]
            else:
                total_normal_errors = torch.cat((total_normal_errors, E[mask]), dim=0)

        total_normal_errors = total_normal_errors.data.cpu().numpy()
        metrics = utils.compute_normal_errors(total_normal_errors)
        utils.log_normal_errors(metrics, where_to_write, first_line='total_iter: {}'.format(total_iter))
        return metrics


# main worker
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # define model
    if args.use_baseline:
        from models.baseline import NNET
    else:
        from models.NNET import NNET
    model = NNET(args)

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    train(model, args, device=args.gpu)


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_dir', default='./experiments', type=str, help='directory to store experiment results')
    parser.add_argument('--exp_name', default='exp00_test', type=str, help='experiment name')
    parser.add_argument('--visible_gpus', default='01', type=str, help='gpu to use')

    # model architecture
    parser.add_argument('--architecture', default='GN', type=str, help='{BN, GN}')
    parser.add_argument("--use_baseline", action="store_true", help='use baseline encoder-decoder (no pixel-wise MLP, no uncertainty-guided sampling')
    parser.add_argument('--sampling_ratio', default=0.4, type=float)
    parser.add_argument('--importance_ratio', default=0.7, type=float)

    # loss function
    parser.add_argument('--loss_fn', default='UG_NLL_ours', type=str, help='{L1, L2, AL, NLL_vMF, NLL_ours, UG_NLL_vMF, UG_NLL_ours}')

    # training
    parser.add_argument('--n_epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--validate_every', default=5000, type=int, help='validation period')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualization period')
    parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
    parser.add_argument("--workers", default=12, type=int, help="Number of workers for data loading")

    # optimizer setup
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--same_lr', default=False, action="store_true", help="Use same LR for all param groups")
    parser.add_argument('--grad_clip', default=0.1, type=float)
    parser.add_argument('--div_factor', default=25.0, type=float, help="Initial div factor for lr")
    parser.add_argument('--final_div_factor', default=10000.0, type=float, help="final div factor for lr")

    # dataset
    parser.add_argument("--dataset_name", default='nyu', type=str, help="{nyu, scannet}")

    # dataset - preprocessing
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_hflip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_random_crop", default=False, action="store_true")

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.num_threads = args.workers
    args.mode = 'train'

    # create experiment directory
    args.exp_dir = args.exp_dir + '/{}/'.format(args.exp_name)
    args.exp_model_dir = args.exp_dir + '/models/'    # store model checkpoints
    args.exp_vis_dir = args.exp_dir + '/vis/'         # store training images
    args.exp_log_dir = args.exp_dir + '/log/'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_model_dir, args.exp_vis_dir, args.exp_log_dir])
    print(args.exp_dir)

    utils.save_args(args, args.exp_log_dir + '/params.txt')  # save experiment parameters
    args.eval_acc_txt = args.exp_log_dir + '/eval_acc.txt'

    # train
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(args.visible_gpus))

    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    args.batch_size_orig = args.batch_size

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)