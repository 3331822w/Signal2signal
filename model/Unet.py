from __future__ import print_function, division
import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size),#scale_factor=2，size=size
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same', bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(U_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        sizes = [[1904, 1639], [1904//2, 1639//2], [1904//4, 1639//4], [1904//8, 1639//8], [1904//16, 1639//16]]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        # self.Conv6 = conv_block(filters[4], filters[5])
        # self.Up6 = up_conv(filters[5], filters[4], sizes[4])
        # self.Up_conv6 = conv_block(filters[5], filters[4])

        self.Up5 = up_conv(filters[4], filters[3], sizes[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        #
        self.Up4 = up_conv(filters[3], filters[2], sizes[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1], sizes[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0], sizes[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding='same')

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        # e2 = self.Conv2(e1)
        # e3 = self.Conv3(e2)
        # e4 = self.Conv_g(e3)
        # e5 = self.Conv_g(e4)
        #
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        #
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)


        # e5 = self.Maxpool4(e4)
        # e5 = self.Conv5(e5)

        # e6 = self.Maxpool5(e5)
        # e6 = self.Conv6(e6)
        # print(e1.shape)
        # print(e2.shape)
        # print(e3.shape)
        # # print(e4.shape)
        # print(e5.shape)

        # d6 = self.Up6(e6)
        # # # d5 = nn.functional.interpolate(d5, size=e4.shape[2:4])
        # d6 = torch.cat((e5, d6), dim=1)
        # d6 = self.Up_conv6(d6)
        # d6 = self.dropout(d6)

        # d5 = self.Up5(e5)#e5/d6
        # # # d5 = nn.functional.interpolate(d5, size=e4.shape[2:4])
        # d5 = torch.cat((e4, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        # d5 = self.dropout(d5)

        d4 = self.Up4(e4)#e4/d5
        # d4 = nn.functional.interpolate(d4, size=e3.shape[2:4])
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        # d4 = self.dropout(d4)

        d3 = self.Up3(d4)
        # d3 = nn.functional.interpolate(d3, size=e2.shape[2:4])
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        # d3 = self.dropout(d3)
        #
        d2 = self.Up2(d3)
        # d2 = nn.functional.interpolate(d2, size=e1.shape[2:4])
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        # d2 = self.dropout(d2)
        # #
        # print(d2.shape)
        # # print(d3.shape)
        # # print(d4.shape)
        # # print(d5.shape)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out

def Unet():
    return U_Net(1, 1)
