import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
import numpy as np


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs/2+0.5]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torchvision.transforms.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
    
def to_var(data, grad=True):
    if torch.cuda.is_available():
        data = data.cuda()
    return Variable(data, requires_grad=grad)

def base_model(r):
    model_out = LeNet5(num_classes=2)

    if torch.cuda.is_available():
        model_out.cuda()
        torch.backends.cudnn.benchmark=True
        
    opti = torch.optim.Adam(model_out.params(),lr=r)
    
    return model_out, opti