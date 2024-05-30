import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torchvision
from torchvision import transforms as T
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import cv2
device='cuda'

def load_model(device, model_name='resnet50'):
    if model_name == 'resnet50':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    if model_name == 'resnet34':
        return torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
    
def cuda_checking():
    pass


# https://github.com/developer0hye/PyTorch-Deformable-Convolution-v2/blob/main/dcn.py
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        assert type(kernel_size) == tuple or type(kernel_size) == int
        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        return torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )

def load_image(img_path, tensor=True):
    img = Image.open(img_path).convert("RGB")
    img_to_tensor = transforms.ToTensor()
    return img_to_tensor(img)

def tensor_to_img(tensor_img):
    return T.ToPILImage()(tensor_img[0,:,:].to('cpu'))

# https://www.digitalocean.com/community/tutorials/how-to-detect-and-extract-faces-from-an-image-with-opencv-and-python
def draw_rectangle(img_path, points_list):
    img = image = cv2.imread(img_path)
    for points in points_list:
        x = points[0]
        y = points[1]
        w = points[2]
        h = points[3]
        cv2.rectangle(img, (x, y), (x+w,y+h), color=(0,255,0), thickness=2)
    return img

# https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
