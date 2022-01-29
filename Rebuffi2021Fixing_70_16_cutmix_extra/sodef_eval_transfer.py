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
import torch.nn as nn
import torch.nn.functional as F
import geotorch
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
    return parser.parse_args(args=[])


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

model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf').to(device)



test_loader = test_loader

l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)


##### here we split the set to multi servers and gpus to speed up the test. otherwise it is too slow.
    
iii = 0

x_test = x_test[1000*iii:1000*(iii+1),...]
y_test = y_test[1000*iii:1000*(iii+1),...]



print('run_standard_evaluation_individual', 'Linf')
print(x_test.shape)


epsilon = 8 / 255.
adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')

X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=64)




torch.save(X_adv, 'x_adv_'+str(iii)+'.pt')
torch.save(y_test, 'y_test'+str(iii)+'.pt')




model.logits = Identity()
fc_features = MLP_OUT_ORTH1024()
odefunc = ODEfunc_mlp(0)
odefeature_layers = ODEBlock(odefunc) 
odefc_layers = MLP_OUT_LINEAR()
model_dense = nn.Sequential( odefeature_layers, odefc_layers).to(device)


new_model = nn.Sequential(model, fc_features, model_dense).to(device)
new_model.load_state_dict(statedic_temp)

print(new_model)

total_correct = 0
torch.cuda.empty_cache()

xadv = X_adv.to(device)
target_class = y_test


predicted_class_total = []

batch = 50
for ijk in range(0, 1000//batch):
    x = xadv[batch*ijk:batch*(ijk+1),...]
    predicted_class = np.argmax(new_model(x).cpu().detach().numpy(), axis=1)
    predicted_class_total.append(predicted_class)
    torch.cuda.empty_cache()

predicted_class_total = np.concatenate(predicted_class_total, axis=0 )

# print(target_class)
# print("==========")
# print(predicted_class)
print("transfer attack acc using adv examples generated from original pretrained model without sodef: ", sum(target_class.numpy()== predicted_class_total)/1000)

