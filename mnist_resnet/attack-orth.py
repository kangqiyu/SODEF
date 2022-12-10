import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import geotorch
import math
import torchvision
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from models import *
from model import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
device = torch.device('cuda:0')
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
endtime = 5
timescale = 1
layernum = 0
fc_dim = 64

# folder_savemodel = './EXP/MNIST_resnet0'
folder_savemodel = './EXP/orth_MNIST_resnet_final'


folder = './EXP/orth_resnetfct5_15/model.pth'
# folder = './EXP/resnetfc20_relu_final/model.pth'


act = torch.sin 
# act2 = torch.nn.functional.relu
saved = torch.load(folder)
print(folder)

statedic = saved['state_dict']
args = saved['args']
args.tol = 1e-5

f_coeffi = -1

from torchdiffeq import odeint_adjoint as odeint
# Step 0: Define the neural network model, return logits instead of activation in forward method
class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
#
    def forward(self, t, x):
        return self._layer(x)
# class ODEfunc_mlp(nn.Module):

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 2*fc_dim)
#         self.act1 = act2
#         self.fc2 = ConcatFC(2*fc_dim, 4*fc_dim)
#         self.act2 = act2
#         self.fc3 = ConcatFC(4*fc_dim, fc_dim)
#         self.act3 = act2
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = f_coeffi*self.fc2(t, out)
#         out = self.act2(out)
#         out = f_coeffi*self.fc3(t, out)
#         out = self.act3(out)
#         return out 

# class ODEfunc_mlp(nn.Module):

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 512)
#         self.act1 = act
#         self.fc2 = ConcatFC(512, 512)
#         self.act2 = act
#         self.fc3 = ConcatFC(512, fc_dim)
#         self.act3 = act
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = f_coeffi*self.fc2(t, out)
#         out = self.act2(out)
#         out = f_coeffi*self.fc3(t, out)
#         out = self.act3(out)
#         return out
# class ODEfunc_mlp(nn.Module): #dense_resnet_relu0

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 2*fc_dim)
#         self.act1 = act2
#         self.fc2 = ConcatFC(2*fc_dim, 4*fc_dim)
#         self.act2 = act2
#         self.fc3 = ConcatFC(4*fc_dim, fc_dim)
#         self.act3 = act
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
# #         out = self.act1(out)
#         out = f_coeffi*self.fc2(t, out)
# #         out = self.act2(out)
#         out = f_coeffi*self.fc3(t, out)
#         out = self.act3(out)
#         return out 
    
    
class ODEfunc_mlp(nn.Module): #dense_resnet_relu1,2,7

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(fc_dim, fc_dim)
        self.act1 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = f_coeffi*self.fc1(t, x)
        out = self.act1(out)
        return out 


# class ODEfunc_mlp(nn.Module): #dense_resnet_relu3

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 8*fc_dim)
#         self.act1 = act
#         self.fc2 = ConcatFC(8*fc_dim, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out 
# class ODEfunc_mlp(nn.Module): #dense_resnet_relu4,6

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 16*fc_dim)
#         self.act1 = act
#         self.fc2 = ConcatFC(16*fc_dim, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out 
# class ODEfunc_mlp(nn.Module): #dense_resnet_relu5

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 2048)
#         self.act1 = act
#         self.fc2 = ConcatFC(2048, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out    


# class ODEfunc_mlp(nn.Module): #8

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 32)
#         self.act1 = act
#         self.fc2 = ConcatFC(32, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out   

# class ODEfunc_mlp(nn.Module): #9

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 8)
#         self.act1 = act
#         self.fc2 = ConcatFC(8, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out    

# class ODEfunc_mlp(nn.Module): #9,10

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 8)
#         self.act1 = act
#         self.fc2 = ConcatFC(8, fc_dim)
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = f_coeffi*self.fc1(t, x)
#         out = self.act1(out)
#         out = self.fc2(t, out)
#         return out        
# class ODEfunc_mlp(nn.Module): #dense_resnet_relu11

#     def __init__(self, dim):
#         super(ODEfunc_mlp, self).__init__()
#         self.fc1 = ConcatFC(fc_dim, 4*fc_dim)
#         self.act1 = torch.tanh
#         self.fc2 = ConcatFC(4*fc_dim, fc_dim)
#         self.act2 = torch.sin 
#         self.nfe = 0

#     def forward(self, t, x):
#         self.nfe += 1
#         out = self.fc1(t, x)
#         out = self.act1(out)
#         out = f_coeffi*self.fc2(t, out)
#         out = self.act2(out)
#         return out      
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

# class MLP_IN(nn.Module):

#     def __init__(self):
#         super(MLP_IN, self).__init__()
#         self.fc0 = nn.Linear(784, fc_dim)

#     def forward(self, input_):
#         input_ =  input_.view(-1,784)
#         h1 = F.relu(self.fc0(input_))
#         return h1
class newLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(newLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
#         self.weight = self.weighttemp.T
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ORTHFC(nn.Module):
    def __init__(self, dimin, dimout, bias):
        super(ORTHFC, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=bias)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=bias)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)

class ORTHFC_NOBAIS(nn.Module):
    def __init__(self, dimin, dimout):
        super(ORTHFC_NOBAIS, self).__init__()
        if dimin >= dimout:
            self.linear = newLinear(dimin, dimout,  bias=False)
        else:
            self.linear = nn.Linear(dimin, dimout,  bias=False)
        geotorch.orthogonal(self.linear, "weight")

    def forward(self, x):
        return self.linear(x)
class MLP_OUT_ORT(nn.Module):
    def __init__(self):
        super(MLP_OUT_ORT, self).__init__()
        self.fc0 = ORTHFC(fc_dim, 10, False)#nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1

class MLP_OUT_final(nn.Module):

    def __init__(self):
        super(MLP_OUT_final, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1
    
class MLP_OUT_BALL(nn.Module):

    def __init__(self):
        super(MLP_OUT_BALL, self).__init__()

        self.fc0 = nn.Linear(fc_dim, 10, bias=False)
        self.fc0.weight.data = matrix_temp
    def forward(self, input_):
        h1 = self.fc0(input_)
        return h1  

    
def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    
device = 'cuda' #if torch.cuda.is_available() else 'cpu'
# print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
fc_max = './EXP/fc_maxrowdistance_64_10/ckpt.pth'
saved_temp = torch.load(fc_max,map_location=torch.device('cpu'))
matrix_temp = saved_temp['matrix']
print(matrix_temp.shape)


# Data
print('==> Preparing data..')
def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    testset = datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test)
    return train_loader, test_loader, train_eval_loader, testset

trainloader, testloader, train_eval_loader, testset = get_mnist_loaders(
    False, 128, 1000
)


class fcs(nn.Module):

    def __init__(self,  in_features=512):
        super(fcs, self).__init__()
        self.dropout = 0.1
        self.merge_net = nn.Sequential(nn.Linear(in_features=512,
                                                 out_features=2048),
#                                        nn.ReLU(),
                                       nn.Tanh(),
#                                        nn.Dropout(p=dropout),
                                       nn.Linear(in_features=2048,
                                                 out_features=fc_dim),
                                       nn.Tanh(),
#                                        nn.Sigmoid(),
                                       )

        
    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output


print('==> Building model..')
net = ResNet18()
net.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

net = net.to(device)


# print(net)
net = nn.Sequential(*list(net.children())[0:-1])

#     print(model)
# fcs_temp = fcs()
fcs_temp = MLP_OUT_ORTH512()

# fc_layersa = nn.Linear(fc_dim, 10,  bias=True)
# fc_layersa = MLP_OUT_ORT()
fc_layersa = MLP_OUT_BALL()

model_fea = nn.Sequential(*net, fcs_temp, fc_layersa).to(device)
saved_temp = torch.load(folder_savemodel+'/ckpt.pth')
# saved_temp = torch.load(folder_savemodel+'/ckpt-Copy1.pth')
statedic_temp = saved_temp['net']
model_fea.load_state_dict(statedic_temp)
    
    
odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)] 
fc_layers = [MLP_OUT_final()]
model_dense = nn.Sequential( *feature_layers, *fc_layers).to(device)
statedic = saved['state_dict']
model_dense.load_state_dict(statedic)


class tempnn(nn.Module):
    def __init__(self):
        super(tempnn, self).__init__()
    def forward(self, input_):
        h1 = input_[...,0,0]
        return h1
tempnn_ = tempnn()
model = nn.Sequential(*net, tempnn_,fcs_temp,  *model_dense).to(device)
# model = nn.Sequential(*net, tempnn_,fcs_temp,  fc_layersa).to(device)

model.eval()
print(model)

# Step 2a: Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type="gpu"
)



def accuracy_PGD(classifier, dataset_loader):
    attack = ProjectedGradientDescent(classifier, eps=0.3, max_iter=20)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x)
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def accuracy_clean(classifier, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
#         predicted_class = np.argmax(predictions.cpu().detach().numpy(), axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def accuracy_CW(classifier, dataset_loader):
    attack = CarliniL2Method(classifier, confidence=1, max_iter=100)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x)
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

def accuracy_FGSM(classifier, dataset_loader):
    attack = FastGradientMethod(estimator=classifier, eps=0.3)
    total_correct = 0
    for x, y in dataset_loader:
#         x = x.to(device)
        x = attack.generate(x=x)
        predictions = classifier.predict(x)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(predictions, axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)

# def accuracy_FGSM(classifier, dataset_loader):
#     attack = FastGradientMethod(estimator=classifier, eps=0.3)
#     total_correct = 0
#     error=0
#     corr = 0
#     for x, y in dataset_loader:
# #         x = x.to(device)
#         try:
#             x = attack.generate(x=x)
#             predictions = classifier.predict(x)
#             y = one_hot(np.array(y.numpy()), 10)
#             target_class = np.argmax(y, axis=1)
#             predicted_class = np.argmax(predictions, axis=1)
#             total_correct += np.sum(predicted_class == target_class)
#             corr = corr+1
#         except:
#             error=error+1
#             continue
#     print('num err: ', error)
#     print('actual acc: ', total_correct/corr)
#     return total_correct / len(dataset_loader.dataset)

print('********************')
# for i in range(100):
#     print(torch.min(testset.__getitem__(i)[0]))
#     print(torch.max(testset.__getitem__(i)[0]))
# print(torch.max(testset[0]))
print(folder, ' time: ', endtime)

class mnist_samples(Dataset):
    def __init__(self, dataset, leng, iid):
        self.dataset = dataset
        self.len = leng
        self.iid = iid
    def __len__(self):
#             return 425
            return self.len

    def __getitem__(self, idx):
        x,y = self.dataset[idx+self.len*self.iid]
        return x,y
test_samples = mnist_samples(testset,1000,7)
# test_loader_samples = DataLoader(test_samples, batch_size=1, shuffle=False, num_workers=2, drop_last=False)
test_loader_samples = DataLoader(test_samples, batch_size=100, shuffle=False, num_workers=2, drop_last=False)

# Step 7: Evaluate the ART classifier on adversarial test examples


accuracy_clean = accuracy_clean(classifier, testloader)
print("Accuracy on benign test examples: {}%".format(accuracy_clean * 100))

accuracy = accuracy_FGSM(classifier, test_loader_samples)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# accuracy = accuracy_CW(classifier, testloader)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


accuracy = accuracy_PGD(classifier, testloader)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

# from robustbench.data import _load_dataset

# # from robustbench.data import load_cifar10
# epsilon = 0.3
# batch_size = 50

# # x_test, y_test = load_cifar10(n_examples=500)
# x_test, y_test = _load_dataset(testset,50)


# from autoattack import AutoAttack
# adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')

# x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size)