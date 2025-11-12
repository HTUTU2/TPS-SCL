import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import argparse
import cv2
from Code.lib.model import ALSOD
from Code.utils.data import test_dataset
import time
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--test_path', type=str, default='', help='test dataset path')
opt = parser.parse_args()

# Define multiple model weight paths
weight_paths = [
    'path/SPNet_epoch_best.pth',
]

# Load the model once
model = ALSOD()
model.cuda()

test_datasets = {
    'UVT2000': 'png',
    # 'VT821_unalign': 'jpg',
    # 'VT1000_unalign': 'jpg',
    # 'VT5000-Test_unalign': 'png',
    # 'VT5000/Test': 'png',
    # 'VT1000': 'jpg',
    # 'VT821': 'jpg',
    'UVT20K/Test': 'png',
}

for weight_path in weight_paths:
    # Load model weights
    state_dict = torch.load(weight_path, map_location='cuda:0')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Extract model name for saving results
    model_name = os.path.splitext(os.path.basename(weight_path))[0]

    for dataset, gt_format in test_datasets.items():
        # Create save path with model name
        save_path = os.path.join('./test_maps', model_name, dataset)
        os.makedirs(save_path, exist_ok=True)

        # Setup dataset paths
        image_root = os.path.join(opt.test_path, dataset, 'RGB/')
        gt_root = os.path.join(opt.test_path, dataset, 'GT/')
        depth_root = os.path.join(opt.test_path, dataset, 'T/')
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
        img_num = len(test_loader)

        total_mae = 0.0
        with tqdm(total=img_num, desc=f'Testing {model_name} on {dataset}', unit='img') as pbar:
            for i in range(test_loader.size):
                image, gt, depth, name, _ = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                depth = depth.cuda()

                with torch.no_grad():
                    res = model(image, depth)

                # Handle tuple output
                if isinstance(res, tuple):
                    res = res[0]

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                # Calculate MAE
                mae = np.mean(np.abs(res - gt))
                total_mae += mae

                # Save results with appropriate extension
                save_name = os.path.splitext(name)[0] + '.' + gt_format
                cv2.imwrite(os.path.join(save_path, save_name), res * 255)
                pbar.update(1)

        avg_mae = total_mae / img_num
        print(f'{model_name} on {dataset} - MAE: {avg_mae:.4f}')