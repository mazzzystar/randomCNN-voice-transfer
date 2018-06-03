import torch
import torch.nn as nn
from torch.autograd import Variable
cuda = True if torch.cuda.is_available() else False

N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32


class RandomCNN(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()

        # 2-D CNN
        self.conv1 = nn.Conv2d(1, OUT_CHANNELS, kernel_size=(3, 1), stride=1, padding=0)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the random parameters to be constant.
        weight = torch.randn(self.conv1.weight.data.shape)
        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias = torch.zeros(self.conv1.bias.data.shape)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_delta):
        out = self.LeakyReLU(self.conv1(x_delta))
        return out


"""
a_random = Variable(torch.randn(1, 1, 257, 430)).float()
model = RandomCNN()
a_O = model(a_random)
print(a_O.shape)
"""