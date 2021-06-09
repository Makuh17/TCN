from numpy.lib import index_tricks
import torch.nn.functional as F
import torch
from torch import nn


#from https://github.com/vsitzmann/deepvoxels/blob/master/losses.py 
class DiscriminatorLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(DiscriminatorLoss, self).__init__()
        self.eps = 1e-9

    def __call__(self, input, target_is_real):
        if target_is_real:
            return -1.*torch.mean(torch.log(input + self.eps))
        else:
            return -1.*torch.mean(torch.log(1 - input + self.eps))

class Discriminator(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size):
        super(Discriminator, self).__init__()
        modules = []
        in_filters = input_channels
        for out_filters in num_filters:
            modules.append(nn.Sequential(nn.Conv1d(in_filters,out_filters, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2)), nn.LeakyReLU(0.2)))
            in_filters = out_filters
        modules.append(nn.Sequential(nn.Conv1d(in_filters,1,1,1),nn.AdaptiveAvgPool1d(1),nn.Sigmoid()))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        res = self.model(input)
        return res

class DiscriminatorSmall(nn.Module):
    def __init__(self, input_channels, kernel_size):
        super(DiscriminatorSmall, self).__init__()
        num_filters = [30, 20, 10]
        modules = []
        in_filters = input_channels
        for out_filters in num_filters:
            modules.append(nn.Sequential(nn.Conv1d(in_filters,out_filters, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2)), nn.LeakyReLU(0.2)))
            in_filters = out_filters
        modules.append(nn.Sequential(nn.Conv1d(in_filters,kernel_size,1,1),nn.AdaptiveAvgPool1d(1),nn.Sigmoid()))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        res = self.model(input)
        return res

class DiscriminatorMinimal(nn.Module):
    def __init__(self, input_channels, kernel_size):
        super(DiscriminatorMinimal, self).__init__()
        num_filters = [5, 2]
        modules = []
        in_filters = input_channels
        for out_filters in num_filters:
            modules.append(nn.Sequential(nn.Conv1d(in_filters,out_filters, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2)), nn.LeakyReLU(0.2)))
            in_filters = out_filters
        modules.append(nn.Sequential(nn.Conv1d(in_filters,kernel_size,1,1),nn.AdaptiveAvgPool1d(1),nn.Sigmoid()))
        self.model = nn.Sequential(*modules)

    def forward(self, input):
        res = self.model(input)
        return res