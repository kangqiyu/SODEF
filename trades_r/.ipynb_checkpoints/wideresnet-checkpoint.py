import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activation='ReLU', softplus_beta=1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if activation == 'ReLU':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
            print('R')
        elif activation == 'Softplus':
            self.relu1 = nn.Softplus(beta=softplus_beta, threshold=20)
            self.relu2 = nn.Softplus(beta=softplus_beta, threshold=20)
            print('S')
        elif activation == 'GELU':
            self.relu1 = nn.GELU()
            self.relu2 = nn.GELU()
            print('G')
        elif activation == 'ELU':
            self.relu1 = nn.ELU(alpha=1.0, inplace=True)
            self.relu2 = nn.ELU(alpha=1.0, inplace=True)
            print('E')

        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activation='ReLU', softplus_beta=1):
        super(NetworkBlock, self).__init__()
        self.activation = activation
        self.softplus_beta = softplus_beta
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate,
                self.activation, self.softplus_beta))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=False, activation='ReLU', softplus_beta=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.normalize = normalize
        #self.scale = scale
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
        elif activation == 'GELU':
            self.relu = nn.GELU()
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
        print('Use activation of ' + activation)

        if self.normalize:
            self.fc = nn.Linear(nChannels[3], num_classes, bias = False)
        else:
            self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not self.normalize:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
            for _, module in self.fc.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc(out)

    
class WideResNet1(nn.Module):
    def __init__(self, fc_dim, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=False, activation='ReLU', softplus_beta=1):
        super(WideResNet1, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.normalize = normalize
        #self.scale = scale
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
        elif activation == 'GELU':
            self.relu = nn.GELU()
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
        print('Use activation of ' + activation)

        if self.normalize:
            self.fc1 = nn.Linear(nChannels[3], fc_dim, bias = False)
        else:
            self.fc1 = nn.Linear(nChannels[3], fc_dim)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not self.normalize:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
            for _, module in self.fc.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc1(out)
    
    
class WideResNet_save(nn.Module):
    def __init__(self, fc_dim, depth=34, num_classes=10, widen_factor=10, dropRate=0.0, normalize=False, activation='ReLU', softplus_beta=1):
        super(WideResNet_save, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.normalize = normalize
        #self.scale = scale
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation=activation, softplus_beta=softplus_beta)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])

        if activation == 'ReLU':
            self.relu = nn.ReLU(inplace=True)
        elif activation == 'Softplus':
            self.relu = nn.Softplus(beta=softplus_beta, threshold=20)
        elif activation == 'GELU':
            self.relu = nn.GELU()
        elif activation == 'ELU':
            self.relu = nn.ELU(alpha=1.0, inplace=True)
        print('Use activation of ' + activation)

        self.fc1 = fcs(fc_dim)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not self.normalize:
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
            for _, module in self.fc.named_modules():
                if isinstance(module, nn.Linear):
                    module.weight.data = F.normalize(module.weight, p=2, dim=1)
        return self.fc1(out)
    
    
    

class fcs(nn.Module):

    def __init__(self,  fc_dim):
        super(fcs, self).__init__()
        self.dropout = 0.1
        self.merge_net = nn.Sequential(nn.Linear(in_features=640,
                                                 out_features=2048),
                                       nn.Tanh(),
                                       nn.Linear(in_features=2048,
                                                 out_features=fc_dim),
                                       nn.Tanh(),
                                       )

        
    def forward(self, inputs):
        output = self.merge_net(inputs)
        return output