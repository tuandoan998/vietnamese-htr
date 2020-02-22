import torch.nn as nn
import torchvision
import torch.nn.functional as F
from collections import OrderedDict

class FE(nn.Module):
    def __init__(self):
        super().__init__()

    def get_cnn(self):
        raise NotImplementedError()
    
    def get_n_features(self):
        raise NotImplementedError()
    
    def forward(self, inputs):
        '''
        :param inputs: [B, C, H, W]
        :returms: [B, C', H', W']
        '''
        return self.get_cnn()(inputs) # [B, C', H', W']

class DenseNetFE(FE):
    def __init__(self, depth, n_blocks, growth_rate):
        super().__init__()
        densenet = torchvision.models.DenseNet(
            growth_rate=growth_rate,
            block_config=[depth]*n_blocks
        )

        self.cnn = densenet.features
        self.n_features = densenet.classifier.in_features

    def get_cnn(self):
        return self.cnn
    
    def get_n_features(self):
        return self.n_features
    
'''
LeNet-5
https://engmrk.com/lenet-5-a-classic-cnn-architecture/
https://github.com/bollakarthikeya/LeNet-5-PyTorch/blob/master/lenet5_gpu.py
https://www.kaggle.com/usingtc/lenet-with-pytorch
'''
class LeNetFE(FE):
    def __init__(self):
        super().__init__()        
        self.cnn = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2)),
            ('conv1', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2)),
        ]))
        self.n_features = 16

    def get_cnn(self):
        return self.cnn

    def get_n_features(self):
        return self.n_features