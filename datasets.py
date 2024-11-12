import numpy as np
import os
import rawpy
import torch
from PIL import Image
from skimage import io, transform
import time
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import get_files
import torch.nn.functional as F


class BaseDataset(Dataset):

    def load_img(self, fname):
        input = io.imread(os.path.join(self.data_path, 'input', fname))
        output = io.imread(os.path.join(self.data_path, 'output', fname))
        input = torch.from_numpy(input.transpose((2, 0, 1)))
        output = torch.from_numpy(output.transpose((2, 0, 1)))
        return input, output

    def load_img_hdr(self, fname):
        input = rawpy.imread(os.path.join(self.data_path, 'inputs', fname))
        # print(f"input_file is {os.path.join(self.data_path, 'inputs', fname)}")
        input = input.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        input = np.asarray(input, dtype=np.float32)
        # Output may not necessarily be .jpg images, so change when necessary.
        output = io.imread(os.path.join(self.data_path, 'targets', fname.split('.')[0] + '.tif'))
        # print(f"output_file is {os.path.join(self.data_path, 'targets', fname.split('.')[0] + '.tif')}")
        output = output.astype(np.float32)
        input = torch.from_numpy(input.transpose((2, 0, 1)))
        output = torch.from_numpy(output.transpose((2, 0, 1)))
        # # Original dng image is too big so down-sample
        # input = resize(input, (int(input.shape[1]/2), int(input.shape[2]/2)))
        # output = resize(output, (int(output.shape[1]/2), int(output.shape[2]/2)))

        return input, output

    def get_resized_size(self, input_tensor):
        # 获取原始图片的比例
        orig_channel, orig_height, orig_width = input_tensor.shape
        ratio = orig_width / orig_height
        # 根据比例确定目标分辨率
        if ratio >= 1:
            new_size = (1080, 1920)
        else:
            new_size = (1920, 1080)
        return new_size


    def __len__(self):
        return len(self.input_paths)


class Train_Dataset(BaseDataset):
    """Class for training images."""

    def __init__(self, params=None):
        self.data_path = params['train_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path, 'inputs'))
        self.input_res = params['input_res']
        self.output_res = params['output_res']
        self.augment = transforms.Compose([
            transforms.RandomCrop(self.output_res),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
        ])
        self.params = params



    def __getitem__(self, idx):
        fname = self.input_paths[idx].split('/')[-1]
        # print(f'{fname=}')
        if self.params['hdr']:
            input, output = self.load_img_hdr(fname)
        else:
            input, output = self.load_img(fname)
        # Check dimensions before crop

        # 根据原始图片的比例调整input和output的大小
        size_input = self.get_resized_size(input)
        size_output = self.get_resized_size(output)
        # print(f'{input.shape=}, {output.shape=}')
        # print(f'{size_input=}, {size_output=}')
        input = F.interpolate(input.unsqueeze(0), size=size_input, mode='bilinear',
                                      align_corners=False).squeeze(0)
        output = F.interpolate(output.unsqueeze(0), size=size_output, mode='bilinear',
                                       align_corners=False).squeeze(0)

        #print(f'after intepolate, {input.shape=}, {output.shape=}')

        if input.shape != output.shape:
            output = torch.rot90(output, k=1, dims=(1, 2))

        assert input.shape == output.shape
        assert self.output_res[0] <= input.shape[2]
        assert self.output_res[1] <= input.shape[1]
        # Crop
        inout = torch.cat([input,output],dim=0)
        inout = self.augment(inout)

        full = inout[:3,:,:]
        low = resize(full, (self.input_res, self.input_res), Image.NEAREST)
        output = inout[3:,:,:]
        return low, full, output


class Eval_Dataset(BaseDataset):
    """Class for validation images."""

    def __init__(self, params=None):
        self.data_path = params['eval_data_dir']
        self.input_paths = get_files(os.path.join(self.data_path, 'inputs'))
        self.input_res = params['input_res']
        self.params = params

    def __getitem__(self, idx):
        fname = self.input_paths[idx].split('/')[-1]
        if self.params['hdr']:
            full, output = self.load_img_hdr(fname)
        else:
            full, output = self.load_img(fname)

        # print(f'Eval dataset, {full.shape=},{output.shape=}')
        low = resize(full, (self.input_res, self.input_res), Image.NEAREST)

        # 根据原始图片的比例调整input和output的大小
        size_full = self.get_resized_size(full)
        size_output = self.get_resized_size(output)
        # print(f'{input.shape=}, {output.shape=}')
        # print(f'{size_input=}, {size_output=}')
        full = F.interpolate(full.unsqueeze(0), size=size_full, mode='bilinear',
                              align_corners=False).squeeze(0)
        output = F.interpolate(output.unsqueeze(0), size=size_output, mode='bilinear',
                               align_corners=False).squeeze(0)
        # print(f'Eval dataset, {full.shape=}, {output.shape=}')
        # print(f'After resize, {low.shape=}')
        return low, full, output