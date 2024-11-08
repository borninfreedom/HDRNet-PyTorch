import numpy as np
import os
import time
import torch
import torch.nn as nn
from argparse import ArgumentParser
from datasets import Train_Dataset, Eval_Dataset
from models import HDRnetModel
from torch.optim import Adam, lr_scheduler
from torchvision.transforms.functional import vflip, hflip
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import psnr, print_params, load_train_ckpt, save_model_stats, plot_per_check, AvgMeter
import torch.nn.functional as F


def train(params, train_loader, valid_loader, model,device):
    # Optimization
    optimizer = Adam(model.parameters(), params['learning_rate'], weight_decay=1e-8)
    # # Learning rate adjustment
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
    #     patience=params['epochs']/4, factor=0.5, verbose=True)

    # Loss function
    criterion = nn.MSELoss()

    # Training
    train_loss_meter = AvgMeter()
    train_psnr_meter = AvgMeter()
    stats = {'train_loss': [],
             'train_psnr': [],
             'valid_psnr': []}
    iteration = 0
    old_time = time.time()
    for epoch in range(params['epochs']):
        for batch_idx, (low, full, target) in enumerate(train_loader):
            # if batch_idx==0:
                # print(f'{low.shape=}')
                # print(f'{full.shape=}')
                # print(f'{target.shape=}')
            iteration += 1
            model.train()

            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)

            output = model(low, full)
            # print(f'{torch.max(output)=}')
            # print(f'{torch.min(output)=}')
            # print(f'{torch.max(target)=}')
            # print(f'{torch.min(target)=}')

            # 因为要用y指导x,所以求x的对数概率，y的概率
            # logp_output = F.log_softmax(output, dim=-1)
            # p_target = F.softmax(target, dim=-1)
            #
            # kl_sum = F.kl_div(logp_output, p_target, reduction='sum')
            # kl_mean = F.kl_div(logp_output, p_target, reduction='mean')
            #
            # print(f'{kl_sum=}, {kl_mean=}')

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % params['summary_interval'] == 0:
                train_loss_meter.update(loss.item())
                train_psnr = psnr(output, target).item()
                train_psnr_meter.update(train_psnr)
                new_time = time.time()
                print('[%d/%d] Iteration: %d | Loss: %.4f | PSNR: %.4f | Time: %.2fs' %
                        (epoch+1, params['epochs'], iteration, loss, train_psnr, new_time-old_time))
                old_time = new_time

        if epoch % params['ckpt_interval'] == 0:
            stats['train_loss'].append(train_loss_meter.avg)
            train_loss_meter.reset()
            stats['train_psnr'].append(train_psnr_meter.avg)
            train_psnr_meter.reset()
            valid_psnr = eval(params, valid_loader, model, device,epoch)
            stats['valid_psnr'].append(valid_psnr)
            # plot_per_check(params['stats_dir'], 'Train loss', stats['train_loss'], 'Training loss')
            # plot_per_check(params['stats_dir'], 'Train PSNR', stats['train_psnr'], 'PSNR (dB)')
            # plot_per_check(params['stats_dir'], 'Valid PSNR', stats['valid_psnr'], 'PSNR (dB)')
            ckpt_fname = "epoch_" + str(epoch)+'_iter_' + str(iteration) + ".pt"
            save_model_stats(model, params, ckpt_fname, stats)


def eval(params, valid_loader, model, device,epoch):
    model.eval()
    psnr_meter = AvgMeter()
    with torch.no_grad():
        for batch_idx, (low, full, target) in enumerate(valid_loader):
            low = low.to(device)
            full = full.to(device)
            target = target.to(device)

            # Normalize to [0, 1] on GPU
            if params['hdr']:
                low = torch.div(low, 65535.0)
                full = torch.div(full, 65535.0)
            else:
                low = torch.div(low, 255.0)
                full = torch.div(full, 255.0)
            target = torch.div(target, 255.0)

            output = model(low, full)
            # print(f'{torch.max(low)=}, {torch.min(low)=}')
            # print(f'{torch.max(full)=}, {torch.min(full)=}')
            # print(f'{torch.max(output)=}, {torch.min(output)=}')
            # print(f'{torch.max(target)=}, {torch.min(target)=}')
            save_folder=os.path.join(params['eval_out'],f'epoch_{epoch}')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_image(output, os.path.join(save_folder, str(batch_idx)+'.png'))
            eval_psnr = psnr(output, target).item()
            psnr_meter.update(eval_psnr)

    print ("Validation PSNR: ", psnr_meter.avg)
    return psnr_meter.avg


def parse_args():
    parser = ArgumentParser(description='HDRnet training')
    # Training, logging and checkpointing parameters
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--mps', action='store_true', help='Use mps')
    parser.add_argument('--ckpt_interval', default=2, type=int, help='Interval for saving checkpoints, unit is iteration')
    parser.add_argument('--ckpt_dir', default='./ckpts', type=str, help='Checkpoint directory')
    parser.add_argument('--stats_dir', default='./stats', type=str, help='Statistics directory')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
    parser.add_argument('--summary_interval', default=10, type=int)

    # Data pipeline and data augmentation
    parser.add_argument('--batch_size', default=4, type=int, help='Size of a mini-batch')
    parser.add_argument('--train_data_dir', type=str, required=True, help='Dataset path')
    parser.add_argument('--eval_data_dir', default=None, type=str, help='Directory with the validation data.')
    parser.add_argument('--eval_out', default='./outputs', type=str, help='Validation output path')
    parser.add_argument('--hdr', action='store_true', help='Handle HDR image')

    # Model parameters
    parser.add_argument('--batch_norm', action='store_true', help='Use batch normalization')
    parser.add_argument('--input_res', default=256, type=int, help='Resolution of the down-sampled input')
    parser.add_argument('--output_res', default=(1024, 1024), type=int, nargs=2, help='Resolution of the guidemap/final output')

    return parser.parse_args()


if __name__ == '__main__':

    # Parse training parameters
    params = vars(parse_args())
    print_params(params)

    if params['cuda']:
        print('Using CUDA')
        # Random seeds
        seed = 1024
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # Random seeds
        seed = 1024
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        torch.manual_seed(seed)
        #torch.cuda.manual_seed(seed)



    # Folders
    os.makedirs(params['ckpt_dir'], exist_ok=True)
    os.makedirs(params['stats_dir'], exist_ok=True)
    os.makedirs(params['eval_out'], exist_ok=True)

    # Dataloader for training
    train_dataset = Train_Dataset(params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    # Dataloader for validation
    valid_dataset = Eval_Dataset(params)
    valid_loader = DataLoader(valid_dataset, batch_size=1)

    # Model for training
    model = HDRnetModel(params)
    print(f'{model=}')
    # load_train_ckpt(model, params['ckpt_dir'])
    if params['cuda']:
        device = torch.device("cuda")
    elif params['mps']:
        device = torch.device('mps')
    else:
        device = torch.device("cpu")
    print(f'{device=}')
    model.to(device)

    train(params, train_loader, valid_loader, model,device=device)
