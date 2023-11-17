import torch
import torch.nn.functional as F


def normal_init(m, mean, std):
    if isinstance(m, torch.nn.ConvTranspose2d) or isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class RoughnessModel(torch.nn.Module):
    def __init__(self, d=16, in_channels=3, k_size=4):
        super(RoughnessModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, d, kernel_size=k_size, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(d, d * 2 ,kernel_size= k_size, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(d * 2, d * 2, kernel_size=k_size, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(d * 2, d * 2, kernel_size=k_size, stride=2, padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(d * 2, d, kernel_size=k_size, stride=2, padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(d, d // 2, kernel_size=k_size, stride=2, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(d // 2, 1, kernel_size=k_size, stride=2, padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(1, 1, kernel_size=k_size, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        y = self.deconv1(x)
        y = self.deconv2(y)
        y = self.deconv3(y)
        y = self.deconv4(y)
        return y
