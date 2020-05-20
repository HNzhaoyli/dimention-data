import torchvision.models as models
import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import init
from torch.nn import functional as F


confient = nn.Sequential(nn.Conv2d(2,3,1,1,padding=0),#六个线性分类器
                         nn.Softmax(dim=1))
GAP = nn.AdaptiveAvgPool2d((1,1))
x = torch.rand(2,2,2,3)
y = confient(x) # 大小(2,3,2,3)即3张概率图
print(y)
print('!'*50)
print('y[0]',y[0])
print(y[0].size())#(3 2 3)
print("@"*30)
print(y[:,1,:,:])
print(y[:,1,:,:].size())
