import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class Conv1dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        #self.cut_last_element = (kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1)
        #self.padding = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        #self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding + 1,stride=stride,dilation=dilation,)
        self.cut_last_element = False
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,stride=stride,dilation=dilation,)
    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)
class PhaseNetLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.classes = 3
        self.kernel_size = 7
        self.stride = 4
        self.activation = torch.relu

        self.inc = nn.Conv1d(self.in_channels, 8, 1)
        self.in_bn = nn.BatchNorm1d(8)

        self.conv1 = Conv1dSame(8, 11, self.kernel_size, self.stride)
        self.bnd1 = nn.BatchNorm1d(11)

        self.conv2 = Conv1dSame(11, 16, self.kernel_size, self.stride)
        self.bnd2 = nn.BatchNorm1d(16)

        self.conv3 = Conv1dSame(16, 22, self.kernel_size, self.stride)
        self.bnd3 = nn.BatchNorm1d(22)

        self.conv4 = Conv1dSame(22, 32, self.kernel_size, self.stride)
        self.bnd4 = nn.BatchNorm1d(32)

        self.up1 = nn.ConvTranspose1d(
            32, 22, self.kernel_size, self.stride, padding=self.conv4.padding, output_padding=self.stride-1, 
        )
        self.bnu1 = nn.BatchNorm1d(22)

        self.up2 = nn.ConvTranspose1d(44,16,
            self.kernel_size,
            self.stride,
            padding=self.conv3.padding,
            output_padding=self.stride-1,
        )
        self.bnu2 = nn.BatchNorm1d(16)

        self.up3 = nn.ConvTranspose1d(
            32, 11, self.kernel_size, self.stride, padding=self.conv2.padding, output_padding=self.stride-1, 
        )
        self.bnu3 = nn.BatchNorm1d(11)

        self.up4 = nn.ConvTranspose1d(22, 8, self.kernel_size, self.stride, padding=3, output_padding=self.stride-1, )
        self.bnu4 = nn.BatchNorm1d(8)

        self.out = nn.ConvTranspose1d(16, self.classes, 1)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x_in = self.activation(self.in_bn(self.inc(x)))
        x1 = self.activation(self.bnd1(self.conv1(x_in)))
        x2 = self.activation(self.bnd2(self.conv2(x1)))
        x3 = self.activation(self.bnd3(self.conv3(x2)))
        x4 = self.activation(self.bnd4(self.conv4(x3)))

        x = torch.cat([self.activation(self.bnu1(self.up1(x4))), x3], dim=1)
        x = torch.cat([self.activation(self.bnu2(self.up2(x))), x2], dim=1)
        x = torch.cat([self.activation(self.bnu3(self.up3(x))), x1], dim=1)
        x = torch.cat([self.activation(self.bnu4(self.up4(x))), x_in], dim=1)
        x = self.out(x)
        oc = self.softmax(x) 
        return oc   