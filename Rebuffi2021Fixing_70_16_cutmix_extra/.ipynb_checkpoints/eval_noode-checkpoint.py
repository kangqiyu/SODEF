import argparse
import copy
import logging
import os
import sys
sys.stdout.flush()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from autoattack import AutoAttack
import torch.utils.data as data
import math
from utils_plus import get_loaders

device = torch.device('cuda:0') 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

args = get_args()
trainloader, test_loader, train_eval_loader, test_dataset = get_loaders(args.data_dir, args.batch_size)



from robustbench import load_model

model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')


new_model = model.to(device)



test_loader = test_loader

l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)


# #### here we split the set to multi servers and gpus to speed up the test. otherwise it is too slow.

iii = 0

x_test = x_test[1000*iii:1000*(iii+1),...]
y_test = y_test[1000*iii:1000*(iii+1),...]



print('run_standard_evaluation_individual', 'Linf')
print(x_test.shape)


epsilon = 8 / 255.
adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')




X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=64)

# X_adv = adversary.run_standard_evaluation_individual(x_test, y_test, bs=64)

