import argparse
import copy
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
from temp_util import progress_bar 
from model import *
from utils import *

from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)


device = torch.device('cuda:0') 



robust_feature_savefolder = './EXP/CIFAR10_resnet_Nov_1'
train_savepath='./data/CIFAR10_train_resnetNov1.npz'
test_savepath='./data/CIFAR10_test_resnetNov1.npz'
    
ODE_FC_save_folder = './EXP/CIFAR10_resnet_Nov_1'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()




def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        
        
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

        
def save_training_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
#     print(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
#         print(x.shape)
        
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
#     print(x_save.shape)
    
    np.savez(train_savepath, x_save=x_save, y_save=y_save)


def save_testing_feature(model, dataset_loader):
    x_save = []
    y_save = []
    modulelist = list(model)
    layernum = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y_ = np.array(y.numpy())
        
        for l in modulelist[0:2]:
              x = l(x)
        xo = x
        x_ = xo.cpu().detach().numpy()
        x_save.append(x_)
        y_save.append(y_)
        
    x_save = np.concatenate(x_save)
    y_save = np.concatenate(y_save)
    
    np.savez(test_savepath, x_save=x_save, y_save=y_save)
    

args = get_args()
nepochs_save_robustfeature = 5
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


trainloader, testloader, train_eval_loader, _ = get_loaders(args.data_dir, args.batch_size)


from robustbench import load_model

robust_backbone = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf')


robust_backbone.logits = Identity()
robust_backbone_fc_features = MLP_OUT_ORTH1024()


"""
To speed up the training, we will separately train ODE block and final FC layer.
"""


################################################ Phase 1, save robust feature from backbone ################################################

fc_layers_phase1 = MLP_OUT_BALL()
for param in fc_layers_phase1.parameters():
    param.requires_grad = False
net_save_robustfeature = nn.Sequential(robust_backbone, robust_backbone_fc_features, fc_layers_phase1).to(device)
for param in robust_backbone.parameters():
    param.requires_grad = False

    
print(net_save_robustfeature)
net_save_robustfeature = net_save_robustfeature.to(device)

data_gen = inf_generator(trainloader)
batches_per_epoch = len(trainloader)


best_acc = 0  # best test accuracy

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_save_robustfeature.parameters(), lr=1e-1, eps=1e-2, amsgrad=True)
def train_save_robustfeature(epoch):
    print('\nEpoch: %d' % epoch)
    net_save_robustfeature.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
#         print(inputs.shape)

        outputs = net_save_robustfeature(x)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test_save_robustfeature(epoch):
    global best_acc
    net_save_robustfeature.eval()
    test_loss = 0
    correct = 0
    total = 0
#     modulelist = list(net_save_robustfeature)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = net_save_robustfeature(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net_save_robustfeature': net_save_robustfeature.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
        torch.save(state, robust_feature_savefolder+'/ckpt.pth')
        best_acc = acc
        
        save_training_feature(net_save_robustfeature, train_eval_loader)
        print('----')
        save_testing_feature(net_save_robustfeature, testloader)
        print('------------')
        
makedirs(robust_feature_savefolder)


for epoch in range(0, nepochs_save_robustfeature):
    train_save_robustfeature(epoch)
    # break
    test_save_robustfeature(epoch)
    print('save robust feature to ' + robust_feature_savefolder)
    
    
saved_temp = torch.load(robust_feature_savefolder+'/ckpt.pth')
statedic_temp = saved_temp['net_save_robustfeature']
net_save_robustfeature.load_state_dict(statedic_temp)
    


################################################ Phase 2, train ODE block ################################################    

weight_diag = 10
weight_offdiag = 0
weight_f = 0.1

weight_norm = 0
weight_lossc =  0

exponent = 1.0
exponent_off = 0.1 
exponent_f = 50
time_df = 1
trans = 1.0
transoffdig = 1.0
numm = 16


ODE_FC_odebatch = 32
ODE_FC_ode_epoch = 20




def df_dz_regularizer(f, z):
#     print("+++++++++++")
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1],-1)
        if batchijacobian.shape[0]!=batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")
            
        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent*(tempdiag+trans))
        offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
        off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
        regu_offdiag += off_diagtemp

    print('diag mean: ',tempdiag.mean().item())
    print('offdiag mean: ',offdiat.mean().item())
    return regu_diag/numm, regu_offdiag/numm


def f_regularizer(f, z):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    print('tempf: ', tempf.mean().item())
    
    return regu_f

def temp1(f, z, text_file):
    regu_diag = 0.
    regu_offdiag = 0.0
    for ii in np.random.choice(z.shape[0], min(numm,z.shape[0]),replace=False):
        batchijacobian = torch.autograd.functional.jacobian(lambda x: odefunc(torch.tensor(time_df).to(device), x), z[ii:ii+1,...], create_graph=True)
        batchijacobian = batchijacobian.view(z.shape[1],-1)
        if batchijacobian.shape[0]!=batchijacobian.shape[1]:
            raise Exception("wrong dim in jacobian")
        tempdiag = torch.diagonal(batchijacobian, 0)
        regu_diag += torch.exp(exponent*(tempdiag+trans))
        offdiat = torch.sum(torch.abs(batchijacobian)*((-1*torch.eye(batchijacobian.shape[0]).to(device)+0.5)*2), dim=0)
        off_diagtemp = torch.exp(exponent_off*(offdiat+transoffdig))
        regu_offdiag += off_diagtemp

    text_file.write('diag mean: '+str(tempdiag.mean().item())+'\n')
    text_file.write('offdiag mean: '+str(offdiat.mean().item())+'\n')
    return 0
def temp2(f, z, text_file):
    tempf = torch.abs(odefunc(torch.tensor(time_df).to(device), z))
    regu_f = torch.pow(exponent_f*tempf,2)
    text_file.write('tempf: '+str(tempf.mean().item())+'\n')
    return 0


class DensemnistDatasetTrain(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(train_savepath)
        
        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y
class DensemnistDatasetTest(Dataset):
    def __init__(self):
        """
        """
        npzfile = np.load(test_savepath)
        
        self.x = npzfile['x_save']
        self.y = npzfile['y_save']
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,...]
        y = self.y[idx]
            
        return x,y    
    

makedirs(ODE_FC_save_folder)

odefunc = ODEfunc_mlp(0)
feature_layers = ODEBlocktemp(odefunc)
fc_layers = MLP_OUT_LINEAR()
for param in fc_layers.parameters():
    param.requires_grad = False

ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)


train_loader_ODE =  DataLoader(DensemnistDatasetTrain(),
     batch_size=ODE_FC_odebatch,
    shuffle=True, num_workers=2
)
train_loader_ODE__ =  DataLoader(DensemnistDatasetTrain(),
     batch_size=ODE_FC_odebatch,
    shuffle=True, num_workers=2
)

test_loader_ODE =  DataLoader(DensemnistDatasetTest(),
     batch_size=ODE_FC_odebatch,
    shuffle=True, num_workers=2
)


data_gen = inf_generator(train_loader_ODE)
batches_per_epoch = len(train_loader_ODE)

optimizer = torch.optim.Adam(ODE_FCmodel.parameters(), lr=1e-2, eps=1e-3, amsgrad=True)


for itr in range(ODE_FC_ode_epoch * batches_per_epoch):

    optimizer.zero_grad()
    x, y = data_gen.__next__()
    x = x.to(device)

    modulelist = list(ODE_FCmodel)
    y0 = x
    x = modulelist[0](x)
    y1 = x

#         y00 = y1.clone().detach().requires_grad_(True)
    y00 = y0#.clone().detach().requires_grad_(True)

    regu1, regu2  = df_dz_regularizer(odefunc, y00)
    regu1 = regu1.mean()
    regu2 = regu2.mean()
    print("regu1:weight_diag "+str(regu1.item())+':'+str(weight_diag))
    print("regu2:weight_offdiag "+str(regu2.item())+':'+str(weight_offdiag))
    regu3 = f_regularizer(odefunc, y00)
    regu3 = regu3.mean()
    print("regu3:weight_f "+str(regu3.item())+':'+str(weight_f))
    loss = weight_f*regu3 + weight_diag*regu1+ weight_offdiag*regu2
#         loss = weight_f*regu3 


    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()
    

    if itr % batches_per_epoch == 0:
        if itr ==0:
            recordtext = os.path.join(ODE_FC_save_folder, 'output.txt')
            if os.path.isfile(recordtext):
                os.remove(recordtext)
            # continue

        with torch.no_grad():
            if True:#val_acc > best_acc:
                torch.save({'state_dict': ODE_FCmodel.state_dict()}, os.path.join(ODE_FC_save_folder, 'model_'+str(itr // batches_per_epoch)+'.pth'))
            with open(recordtext, "a") as text_file:
                text_file.write("Epoch {:04d}".format(itr // batches_per_epoch)+'\n')

                temp1(odefunc, y00, text_file)
                temp2(odefunc, y00, text_file)

                text_file.close()
    # break

################################################ Phase 3, train final FC ################################################    


ODE_FC_fcbatch = 128
ODE_FC_fc_epoch = 10




feature_layers = ODEBlock(odefunc)
fc_layers = MLP_OUT_LINEAR()
ODE_FCmodel = nn.Sequential(feature_layers, fc_layers).to(device)



# saved = torch.load('./EXP/CIFAR10_resnetNov/dense1_temp/model_.pth')
# statedic = saved['state_dict']
# ODE_FCmodel.load_state_dict(statedic)

for param in odefunc.parameters():
    param.requires_grad = False
for param in robust_backbone_fc_features.parameters():
    param.requires_grad = False
for param in robust_backbone.parameters():
    param.requires_grad = False

new_model_full = nn.Sequential(robust_backbone, robust_backbone_fc_features, ODE_FCmodel).to(device)

# saved_temp = torch.load( './EXP/CIFAR10_resnet_Nov_1/full.pth')
# statedic_temp = saved_temp['state_dict']
# new_model_full.load_state_dict(statedic_temp)


optimizer = torch.optim.Adam(new_model_full.parameters(), lr=1e-2, eps=1e-4, amsgrad=True)

# optimizer = torch.optim.Adam([{'params': odefunc.parameters(), 'lr': 1e-4, 'eps':1e-5,},
#                             {'params': fc_layers.parameters(), 'lr': 1e-2, 'eps':1e-3,}], amsgrad=True)


criterion = nn.CrossEntropyLoss()



# Training
def train(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        x = inputs
#         print(inputs.shape)

        outputs = net(x)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(net, epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
# #     modulelist = list(net)
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             x = inputs
#             outputs = net(x)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
# #         if not os.path.isdir('checkpoint'):
# #             os.mkdir('checkpoint')
# #         torch.save(state, './checkpoint/ckpt.pth')
#         torch.save(state, folder_savemodel+'/ckpt.pth')
#         best_acc = acc
        


    


    
    
use_cuda = True    
alpha = 1.0
###################### option 2 use mixup to boost the robustness ######################
from torch.autograd import Variable
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_mixup(net, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.item()#loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Reg: %.5f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), reg_loss/(batch_idx+1),
                        100.*correct/total, correct, total))
    return (train_loss/batch_idx, reg_loss/batch_idx, 100.*correct/total)



best_acc = 0



for epoch in range(0, ODE_FC_fc_epoch):
    
    train(new_model_full, epoch)
#     train_mixup(new_model_full, epoch)
    
    with torch.no_grad():
        val_acc = accuracy(new_model_full, testloader)
        if val_acc > best_acc:
            torch.save({'state_dict': new_model_full.state_dict()}, os.path.join(ODE_FC_save_folder, 'full.pth'))
            best_acc = val_acc
        print("Epoch {:04d} |  Test Acc {:.4f}".format(epoch,  val_acc))
        
        
        
    # break
    
torch.save({'state_dict': new_model_full.state_dict()}, os.path.join(ODE_FC_save_folder, 'full.pth'))    
# saved_temp = torch.load(os.path.join(ODE_FC_save_folder, 'full.pth'))
# statedic_temp = saved_temp['state_dict']
# new_model_full.load_state_dict(statedic_temp)


################################################ attack ################################################   
testloader = testloader

l = [x for (x, y) in testloader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in testloader]
y_test = torch.cat(l, 0)


##### here we split the set to multi servers and gpus to speed up the test. otherwise it is too slow.
##### if your server is powerful or your have enough time, just use the full dataset directly by commenting out the following.
#############################################    
iii = 0

x_test = x_test[1000*iii:1000*(iii+1),...]
y_test = y_test[1000*iii:1000*(iii+1),...]

#############################################   

print('run_standard_evaluation_individual', 'Linf')
print(x_test.shape)


epsilon = 8 / 255.
adversary = AutoAttack(new_model_full, norm='Linf', eps=epsilon, version='standard')




X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)