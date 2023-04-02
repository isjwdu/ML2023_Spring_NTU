import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision import models

class VGG19_BN(nn.Module):

    def __init__(self, classes_num):                                
        super(VGG19_BN, self).__init__()
        self.vgg19_bn = models.vgg19_bn(weights=None, progress=True)

    def forward(self, x):
        x = self.vgg19_bn(x)
        return x

class EFFICIENTNET_V2_S(nn.Module):

    def __init__(self, classes_num):                                
        super(EFFICIENTNET_V2_S, self).__init__()
        self.efficientnet_v2_s = models.efficientnet_v2_s(weights=None, progress=True)

    def forward(self, x):
        x = self.efficientnet_v2_s(x)
        return x

class SHUFFLENET(nn.Module):

    def __init__(self, classes_num):
        super(SHUFFLENET, self).__init__()
        self.shufflenet_v2_x1_0 = models.shufflenet_v2_x1_0(weights = None, progress = True) 
        
    def forward(self, x):
        x= self.shufflenet_v2_x1_0(x)
        return x

class RESNEXT101(nn.Module):

    def __init__(self, classes_num):
        super(RESNEXT101, self).__init__()
        self.resnext101_64x4d = models.resnext101_64x4d(weights = None, progress = True) 
        
    def forward(self, x):
        x= self.resnext101_64x4d(x)
        return x

