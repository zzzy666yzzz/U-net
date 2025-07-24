import torch
from sympy import false
from torch import nn
from torch.nn import functional as F

class Conv_Block(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Conv_Block,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias = False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()

        )#定义卷积块层

    def forward(self,x):
        return self.layer(x)

class DownSample(nn.Module):  #减少特征图的空间尺寸（高度和宽度），同时通常会增加特征图的通道数
    def __init__(self,channel):
        super(DownSample,self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
    def forward(self,x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample,self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self,x,feature_map):
        up = F.interpolate(x,scale_factor=2, mode='nearest')
        out = self.layer(up)
        return  torch.cat((out,feature_map),dim = 1)

class UNet(nn.Module):
    def __init__(self, num_classes = 21):
        super(UNet,self).__init__()
        self.c1 = Conv_Block(3, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = Conv_Block(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)

    def forward(self, x):
        R1 = self.c1(x)#输入图像经过第一个卷积块 self.c1 进行特征提取
        R2 = self.c2(self.d1(R1)) #R1 经过下采样模块 self.d1 后，再经过第二个卷积块 self.c2
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.out(O4)

if __name__ == '__main__':
    x=torch.randn(2,3,256,256)
    net=UNet()
    output = net(x)
    print(f"输出形状：{output.shape}")
    print(f"每个像素的概率和：{output[0,:,100,100].sum()}")