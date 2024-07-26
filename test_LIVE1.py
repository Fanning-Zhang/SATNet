# -*- coding: utf-8 -*-
"""
This code only contains testing, can output:
> model's parameters & FLOPs (the output of torchinfo: Total mult-adds (G))
> average inference time for a stereo image
> predicted score and bad case
> feature maps visualization (TO DO!)
"""

import os
import time
import random
import sys
import logging
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from scipy import stats
from torch.utils.data import DataLoader
from torchinfo import summary

from satnet import SATNet
from dataset_LIVE import PreProcessDataset

# ----------------------------------------- Configuration ---------------------------------------------- #
parser = argparse.ArgumentParser()
# basic configuration.
parser.add_argument('--workers', type=int, default=8, help='number of data loading cpu workers')
parser.add_argument('--test_batchSize', type=int, default=144, help='test image size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--GPU_id', type=str, default='0', help='which GPU to use')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--network', default=SATNet, help='which network to use')
parser.add_argument('--output_predicted_score', type=bool, default=True, help='enables output predicted score')
parser.add_argument('--bad_case', type=str, default='abs', help='abs or rel (absolute or relative),'
                                                                'calculation methods for bad case')

# path configuration (relative path).
parser.add_argument('--checkpoint_path', type=str, default='./BaseLIVEI/checkpoints/SAT-SE/LIVEI-Epoch-?.pth',
                    help="path of checkpoint to load")
parser.add_argument('--save', type=str, default='./BaseLIVEI/Inference/', help='folder to output log')
parser.add_argument('--exp_name', type=str, default='SAT-SE', help='notation of current experiment for recording')
parser.add_argument('--fea_map_visual', type=str, default='./BaseLIVEI/fea_map_visual/SAT-SE',
                    help='folder of feature map visualization')

cfg = parser.parse_args()

# --------------------------------------- Random seed settings ---------------------------------------- #
if cfg.seed is not None:
    torch.backends.cudnn.benchmark = False      # for cudnn.
    torch.backends.cudnn.deterministic = True

    random.seed(cfg.seed)                       # for python.
    np.random.seed(cfg.seed)                    # for numpy.
    torch.manual_seed(cfg.seed)                 # for CPU.

    torch.cuda.manual_seed(cfg.seed)            # for current GPU.
    # torch.cuda.manual_seed_all(cfg.seed)      # for multiple GPUs.

    def _init_fn():
        np.random.seed(cfg.seed)                # for dataloader.

    print('random seed is: ', cfg.seed)

# ----------------------------------------- Dataset settings ------------------------------------------ #
root_src = '../database/LIVEI'
test_filename = 'test.txt'

# Mean and std are calculated based on LIVE I train l&r.
# TestSet should also use the mean&std calculated by train dataset.
train_l_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.3637327, 0.3423758, 0.27659312],
                                                             std=[0.22781903, 0.2175291, 0.21648386])])
train_r_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.3608473, 0.3396444, 0.27177668],
                                                             std=[0.22896767, 0.21838564, 0.21666355])])

# Parsing and processing .txt files.
test_set = PreProcessDataset(root_src=root_src, list_file=test_filename, train=False,
                             transform_l=train_l_transform, transform_r=train_r_transform)

# Create dataloader.
test_loader = DataLoader(dataset=test_set, batch_size=cfg.test_batchSize, shuffle=False,
                         num_workers=cfg.workers, worker_init_fn=_init_fn(), collate_fn=test_set.collate_fn)

# ------------------------------------------ Device settings ------------------------------------------ #
if torch.cuda.is_available() and not cfg.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Make GPU id number always correspond to the id number specified in the code.
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"

DEVICE = torch.device("cuda:{}".format(cfg.GPU_id) if torch.cuda.is_available() and cfg.cuda else "cpu")
print(DEVICE)

# ----------------------------------------- Network settings ------------------------------------------ #
model = cfg.network()

# load a checkpoint
if os.path.isfile(cfg.checkpoint_path):
    print("=> loading checkpoint '{}'".format(cfg.checkpoint_path))

    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_in_checkpoint = checkpoint['epoch']

    print("=> loaded checkpoint (epoch {})".format(epoch_in_checkpoint))

else:
    print("=> no checkpoint found in '{}'".format(cfg.checkpoint_path))
    
    # stop running when no checkpoint found.
    raise FileNotFoundError     

# network to device and print its total parameters
if cfg.cuda:
    model.to(DEVICE)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Number of model's params: %.4fM" % (total_params / 1e6))

# ----------------------------------------- Record settings ------------------------------------------- #
# Initialize logging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Make directory to output log.
try:
    os.makedirs(cfg.save)
except OSError:
    pass

# Print the feature map size of each model layer.
# 3D-IQA has multiple inputs, so the input_size(C,H,W) is also multiple.
summary(model, input_size=[(64, 3, 40, 40), (64, 3, 40, 40)])

# Make a '.txt' log file to record test details.
exp_dir = cfg.save + '/'
test_log_file = open(exp_dir + '/' + cfg.exp_name + '.txt', 'w')
test_log_file.write("Number of model's params: %.4fM\n" % (total_params / 1e6))
test_log_file.flush()


# ---------------------------------------- Finished configuration ------------------------------------- #
# ------------------------------------------ And start testing --------------------------------------- #
if __name__ == '__main__':

    logging.info('start testing')

    def image_score(patch):
        """Function to calculate score based on different database"""

        outs_image = []
        lines = 0

        for num in range(0, 73):
            # range should be the number of test imgs(LIVEI has 73 test imgs).
            sum0 = 0
            lines = 144 + lines
            for index in range(lines-144, lines):
                sum0 += patch[index]
            scr_image = sum0 / 144
            outs_image.append(scr_image)

        return outs_image

    def calcu_indicators(labels, predicted):
        """Function to calculate 4 indicators"""

        plcc = stats.pearsonr(labels, predicted)[0]
        srocc = stats.spearmanr(labels, predicted)[0]
        rmse = np.sqrt(((predicted - labels) ** 2).mean())
        krocc = stats.stats.kendalltau(predicted, labels)[0]

        return plcc, srocc, rmse, krocc

    # Enter testing mode
    model.eval()

    label_score = []
    predicted_score = []
    time_list = []

    for i, data in enumerate(test_loader):
        # left_image, right_image, label = data
        left_image = data[0].to(DEVICE)
        right_image = data[1].to(DEVICE)

        label = data[2].to(DEVICE)
        label = label.unsqueeze(1)
        label_list = label.cpu().numpy().tolist()
        label_score.append(label_list)

        with torch.no_grad():
            time_start = time.time()
            outputs = model(left_image, right_image)
            torch.cuda.synchronize()
            time_end = time.time()
            time_list.append(time_end - time_start)
            # inference time for testing a stereo image (144 patches)

        predicted_list = outputs.cpu().numpy().tolist()
        predicted_score.append(predicted_list)

    predicted_score = str(predicted_score)
    predicted_score = predicted_score.replace('[', '')
    predicted_score = predicted_score.replace(']', '')
    predicted_score = list(eval(predicted_score))
    outputs_IQ = torch.Tensor(predicted_score)

    label_score = str(label_score)
    label_score = label_score.replace('[', '')
    label_score = label_score.replace(']', '')
    label_score = list(eval(label_score))
    labels_IQ = torch.Tensor(label_score)

    # Calculate image Q from patch Q
    labels_image = np.array(image_score(labels_IQ))
    outputs_image = np.array(image_score(outputs_IQ))

    # Calculate indicators
    PLCC, SROCC, RMSE, KROCC = calcu_indicators(labels_image, outputs_image)

    # Calculate average inference time for a stereo image
    inference_time = np.mean(time_list)

    # Write results in test log file
    test_log_file.write('epoch: %d---PLCC: %.4f  SROCC: %.4f  RMSE: %.4f  KROCC: %.4f\n' %
                        (epoch_in_checkpoint, PLCC, SROCC, RMSE, KROCC))
    test_log_file.write('average inference time for a stereo image: %.4fsec\n' % inference_time)

    test_log_file.flush()

    # Print current indicators and inference time
    print('epoch: %d---PLCC: %.4f  SROCC: %.4f  RMSE: %.4f  KROCC: %.4f  ' %
          (epoch_in_checkpoint, PLCC, SROCC, RMSE, KROCC))
    print('average inference time for a stereo image: %.4fsec' % inference_time)
    
    # Calculate bad case
    if cfg.output_predicted_score:
        if cfg.bad_case == 'abs':
            error_value = (labels_image - outputs_image) ** 2
            bad_case_index = np.argmax(error_value)
            test_log_file.write('bad case calculation: abs\n  label: %.4f  predicted: %.4f' %
                                (labels_image[bad_case_index], outputs_image[bad_case_index]))
            # error_value_sorted = np.sort(error_value)[::-1]     # sort in descending order
            test_log_file.flush()

        elif cfg.bad_case == 'rel':
            error_value = np.abs((labels_image - outputs_image) / labels_image)
            bad_case_index = np.argmax(error_value)
            test_log_file.write('bad case calculation: rel\n  label: %.4f  predicted: %.4f' %
                                (labels_image[bad_case_index], outputs_image[bad_case_index]))
            test_log_file.flush()

        else:
            raise ValueError('bad_case should be abs or rel')

    logging.info('finish testing')

    # Close test log file.
    test_log_file.close()

# ----------------------------------------- Finished All ---------------------------------------------- #
