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

from torch.nn.parameter import Parameter
from autoattack import AutoAttack
import torch.utils.data as data
import math
from model import *
from utils import *
from utils_plus import get_loaders

device = torch.device('cuda:0') 


# saved_temp = torch.load('./EXP/CIFAR10_resnet_Nov_1/full.pth')
saved_temp = torch.load('./EXP/full.pth')
statedic_temp = saved_temp['state_dict']



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()

        

args = get_args()
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


trainloader, test_loader, train_eval_loader, test_dataset = get_loaders(args.data_dir, args.batch_size)



from robustbench import load_model

model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')
model.logits = Identity()


fc_features = MLP_OUT_ORTH1024()
odefunc = ODEfunc_mlp(0)
odefeature_layers = ODEBlock(odefunc) 
odefc_layers = MLP_OUT_LINEAR()
model_dense = nn.Sequential(odefeature_layers, odefc_layers).to(device)


new_model = nn.Sequential(model, fc_features, model_dense).to(device)
new_model.load_state_dict(statedic_temp)



test_loader = test_loader

l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)


##### here we split the set to multi servers and gpus to speed up the test. otherwise it is too slow.
##### if your server is powerful or your have enough time, just use the full dataset directly by commenting out the following.
#############################################    
iii = 9

x_test = x_test[1000*iii:1000*(iii+1),...]
y_test = y_test[1000*iii:1000*(iii+1),...]

#############################################   

print('run_standard_evaluation_individual', 'Linf')
print(x_test.shape)


epsilon = 8 / 255.
adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')




X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=4)
# 
# X_adv = adversary.run_standard_evaluation_individual(x_test, y_test, bs=64)

