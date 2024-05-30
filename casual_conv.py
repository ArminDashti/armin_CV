import torch
import torch.nn as nn

class Casual_Conv1D(nn.Module):
    def __init__(self, receptive_field, kernel_size=2):
        super().__init__()
        self.rf = receptive_field
        self.kernel_size = kernel_size
        self.casual_conv = self.generate_casual_conv()
        
    def generate_casual_conv(self):
        casual_conv = nn.Sequential()
        for i in range(self.rf-1):
            casual_conv.append(nn.Conv1d(1, 1, self.kernel_size))
        return casual_conv
        
    def forward(self, x):
        return self.casual_conv(x)
    
#%%
    
from PIL import Image
from torchvision import transforms as T
import torch
from torch import nn as nn
import numpy as np
import torchvision

conv = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1, dilation=2).to(torch.float64)
relu = nn.Tanh()
x = np.random.randint(0, 256, size=(1,3,200,200))
tenb = x
print(x.max())
x = torch.tensor(x) / 256
x = x.to(torch.float64)
ten = x
x = relu(conv(x))
x.mul(256).int().max()
ten.mul(256).int().max()
#%%
ten