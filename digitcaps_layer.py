import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# in_units : 5
# in_units_size : 128*3*3
# n_units : 55
# unit_size : 128

# input: (batch_size, 128, 5, 3, 3)
# output : (batch_size, 55, 128) - (batch_size, 55)

# 55 → 1とすれば最後の確率を求める式となる。

class DigitCaps_Layer(nn.Module):

    def __init__(self, in_capsule_n=4*4*64, in_capsule_size=4,
                 out_capsule_n=101, out_capsule_size=32):
        super(DigitCaps_Layer, self).__init__()

        self.in_capsule_n = in_capsule_n
        self.in_capsule_size = in_capsule_size
        self.out_capsule_n = out_capsule_n
        self.out_capsule_size = out_capsule_size

        self.W = nn.Parameter(
            torch.randn(in_capsule_n, out_capsule_n, out_capsule_size, in_capsule_size)
        )


    def forward(self, u):
        """
        :param x: (batch_size, out_capsule_size, out_capsule)
                    (batch_size, 5, 128,3,3)
        :return: (batch_size, out_capsule)
        """
        batch_size = u.size(0)
        x = torch.stack([u]*self.out_capsule_n, dim=2)
        W = torch.cat([self.W.unsqueeze(0)] * batch_size, dim=0)
        u_hat = torch.matmul(W, x.unsqueeze(4))

        u_hat_detached = u_hat.detach()

        b_ij = Variable(torch.zeros(self.in_capsule_n, self.out_capsule_n, 1))
        if torch.cuda.is_available():
            b_ij = b_ij.cuda(0)

        iterations = 3
        for iteration in range(iterations):
            c_ij = F.softmax(b_ij.unsqueeze(0), dim=2)
            c_ij = torch.cat([c_ij]*batch_size, dim=0).unsqueeze(4)

            if iteration == iterations-1:
                s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
                v_j = squash(s_j, dim=3)

            else:
                s_j = (c_ij * u_hat_detached).sum(dim=1, keepdim=True)
                v_j = squash(s_j, dim=3)

                u_vj = torch.matmul(
                    u_hat_detached.transpose(3,4), v_j).squeeze(4).mean(dim=0, keepdim=False)
                b_ij = b_ij + u_vj

        return v_j.squeeze(4).squeeze(1)


def squash(s, dim):
    """
    :param s: (batch_size, 55, 128)
    :return: (batch_size, 55)
    """
    norm_sq = torch.sum(s**2, dim=dim, keepdim=True)
    # print(norm_sq.shape)
    norm = torch.sqrt(norm_sq)
    s = (norm/(1+norm_sq)) * (s/norm)
    return s


if __name__=="__main__":
    c = DigitCaps_Layer()
    input = torch.autograd.Variable(torch.randn(1,128,5,3,3))
    for i in range(5):
        m = c(input)
    # x = c._routing(m)
    # print(x.shape)
    print(m.shape)