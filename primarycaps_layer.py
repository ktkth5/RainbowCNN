import torch
import torch.nn as nn
from digitcaps_layer import squash

# in_channel : 2048
# hidden_channel : 512
# out_channel : 128
# n_units : 5


## To make vecor u
class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=4, stride=1, bias=True):
        super(ConvUnit, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def forward(self, x):
        return self.conv(x)

class PrimaryCaps_Layer(nn.Module):

    def __init__(self, in_channels=2048, out_capsule_units=64,
                 out_capsule_n=4*4, out_capsule_size=4,):
        super(PrimaryCaps_Layer, self).__init__()

        self.capsule_units = out_capsule_units
        self.out_capsule_n = out_capsule_n
        self.out_capsule_size = out_capsule_size

        def create_conv_unit(unit_idx):
            unit = ConvUnit(in_channels=in_channels, out_channels=out_capsule_size)
            self.add_module("unit_"+str(unit_idx), unit)
            return unit

        self.conv_units = [create_conv_unit(i) for i in range(out_capsule_units)]


    def forward(self, x):
        """
        :param x: (batch size, 256, 20, 20)
        :return: (batch_size, 5, 128, 3, 3)
                 (batch_size, out_channel, n_units, unit_size)
        """
        batch_size = x.size(0)
        u = []
        for i in range(self.capsule_units):
            u_i = self.conv_units[i](x)
            u_i = u_i.view(batch_size, self.out_capsule_size, -1, 1)
            u.append(u_i)

        u = torch.cat(u, dim=3)
        u = u.view(batch_size, self.out_capsule_size, -1)
        u = u.transpose(1,2)
        u_squashed = squash(u, dim=2)
        # print("Primary Caps")
        # print(u_squashed.shape)
        return u_squashed


if __name__=="__main__":
    c = PrimaryCaps_Layer()
    x = torch.autograd.Variable(torch.randn(2, 2048, 7, 7))
    #print(x)
    u=c(x)
    print(u.shape) # (batch_size,128,5,3,3)