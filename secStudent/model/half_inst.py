import torch
from torch import nn
class half_instance(nn.Module):
    def __init__(self, inc):
        super(half_instance, self).__init__()
        self.half = inc // 2
        self.inst = nn.InstanceNorm2d(self.half)

    def forward(self, x):
        out_1, out_2 = torch.chunk(x, 2, dim=1)
        out = torch.cat([self.inst(out_1), out_2], dim=1)
        return out
