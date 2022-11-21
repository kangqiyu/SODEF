import argparse
import copy
import logging
import os
import sys
sys.stdout.flush()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import torch.optim as optim
import numpy as np
import torch

from torch.nn.parameter import Parameter
from autoattack import AutoAttack
import torch.utils.data as data
import math
from model import *
from utils_plus import get_loaders

device = torch.device('cuda:0') 


saved_temp = torch.load('./EXP/full.pth')
statedic_temp = saved_temp['state_dict']



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x)[0].cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

        

args = get_args()
nepochs = 100
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
model_dense = nn.Sequential( odefeature_layers, odefc_layers).to(device)


SODEF = nn.Sequential(model, fc_features, model_dense).to(device)
SODEF.load_state_dict(statedic_temp)





from robustbench import benchmark

threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
dataset = "cifar10"  # one of {"cifar10", "cifar100", "imagenet"}

model = SODEF
model_name = "SODEF2021Stable"
device = torch.device("cuda:0")

clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  to_disk=True)