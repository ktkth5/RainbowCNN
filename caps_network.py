import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from primarycaps_layer import PrimaryCaps_Layer, ConvUnit
from digitcaps_layer import DigitCaps_Layer

from flip_se_ResNeXt import se_resnext152


class Caps_ResNeXt(nn.Module):

    def __init__(self, in_channels=2048, out_channels=128, capsule_n_=3*3,
                 capsule_size=5, out_capsule_n=55, out_capsule_size=1):
        super(Caps_ResNeXt, self).__init__()

        trained_model = se_resnext152(101)
        lists = list(trained_model.children())[:-2]
        self.model = nn.Sequential(*lists)

        self.primary_layer = PrimaryCaps_Layer()
        self.digit_layer = DigitCaps_Layer()
        self.fc1 = nn.Linear(101*32, 2048)
        self.fc2 = nn.Linear(2048, 101)

    def forward(self, x):
        # x = self.conv0(x)
        x = self.model(x)
        u = self.primary_layer(x)
        # print(u)
        u = self.digit_layer(u)
        u = u.view(-1,101*32)
        u = self.fc1(u)
        u = self.fc2(u)
        # print(u.shape)
        return F.relu(u)


if __name__=="__main__":
    model = Caps_Net()
    model = nn.DataParallel(model).cuda()
    # print(model.primary_layer.parameters())
    # for param in model.digit_layer.parameters():
    #     print(param)
    # print(model.digit_layer.parameters())

    x = torch.autograd.Variable(torch.randn(1, 3, 224, 224))
    out = model(x)
    print(out.shape)