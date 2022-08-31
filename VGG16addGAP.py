import torch.nn.functional
import torch.nn as nn

class fextractBlock(torch.nn.Module):
    def __init__(self):
        super(fextractBlock, self).__init__()
        vgg = []

        # 输入为 224，224，3
        # 第一个卷积部分
        # 112，112，64
        vgg.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第二个卷积部分
        # 56，56，128
        vgg.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第三个卷积部分
        # 28，28，256
        vgg.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第四个卷积部分
        # 14，14，512
        vgg.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第五个卷积部分
        # 7，7，512
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 7,7,32
        # vgg.append(nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1))

        self.VGGconv = nn.Sequential(*vgg)

    def forward(self, x):
        x = self.VGGconv(x)
        return x


class fcBlock(torch.nn.Module):
    def __init__(self):
        super(fcBlock, self).__init__()
        self.GAP = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        batchsize = x.size(0)
        x = self.GAP(x)
        x = x.view(batchsize, -1)
        x = self.fc(x)

        return x

class totalNet(torch.nn.Module):
    def __init__(self):
        super(totalNet, self).__init__()
        self.fextractblock = fextractBlock()
        self.fcblock = fcBlock()

    def forward(self, x):
        x = self.fextractblock(x)
        x = self.fcblock(x)
        return x