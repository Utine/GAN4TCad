import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- Resnet Blocks-----------------------------------------------------

class BasicBlock(nn.Module):
    def __init__(self, dim):
        super(BasicBlock, self).__init__()
        block = []
        block += [nn.Conv3d(dim, dim, 3, 1, 1), nn.BatchNorm3d(dim), nn.ReLU(True)]
        block += [nn.Conv3d(dim, dim, 3, 1, 1), nn.BatchNorm3d(dim)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = F.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, dim, width):
        super(BottleNeck, self).__init__()
        block = []
        block += [nn.Conv3d(dim, width, 1, 1), nn.BatchNorm3d(width), nn.ReLU(True)]
        block += [nn.Conv3d(width, width, 3, 1, 1), nn.BatchNorm3d(width), nn.ReLU(True)]
        block += [nn.Conv3d(width, dim, 1, 1), nn.BatchNorm3d(dim)]
        self.conv_block = nn.Sequential(*block)

    def forward(self, x):
        out = x + self.conv_block(x)
        out = F.relu(out)

        return out


# -----------------------------Discriminator for Ferro-------------------------------------------

class Discrim3DFerro(nn.Module):
    def __init__(self):
        super(Discrim3DFerro, self).__init__()

        # input for D is a combination of input image and fake_volume or real_volume.
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 128, (2, 2, 1), (2, 2, 1), (1, 1, 0)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 256, (2, 2, 2), (2, 2, 2), (0, 0, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(256, 512, (2, 2, 2), (2, 2, 2)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(512, 1, (4, 4, 2), (2, 2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())  # torch.Size([Batch, 128, 16, 16, 6])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([Batch, 256, 8, 8, 4])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([Batch, 512, 4, 4, 2])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([Batch, 1, 1, 1, 1])

        return out


# -----------------------------Discriminator for Channel-------------------------------------


class Discrim3DChan(nn.Module):
    def __init__(self):
        super(Discrim3DChan, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 128, (2, 2, 1), (2, 2, 1), (1, 1, 0)),  # (16, 16, 2)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 256, 2, 2),                           # (8, 8, 1)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(256, 512, (2, 2, 1), (2, 2, 1)),          # (4, 4, 1)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(512, 1, (4, 4, 1), (2, 2, 1)),            # (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class Discrim2DChan(nn.Module):
    def __init__(self):
        super(Discrim2DChan, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 128, 2, 2, 1),         # (16, 16)
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 256, 2, 2),          # (8, 8)
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(256, 512, 2, 2),          # (4, 4)
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(512, 1, 4, 2),            # (1, 1)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

# -----------------------------Generator for Ferro-------------------------------------------


# 2D->3D for ferro encoder-decoder architecture without skip-connections
class NormalGen3DFerro(nn.Module):
    def __init__(self):
        super(NormalGen3DFerro, self).__init__()
        ImageEncoder = [
            nn.Conv2d(2, 16, 6),
            nn.ReLU(),
            nn.Conv2d(16, 32, 6),
            nn.ReLU(),
            nn.Conv2d(32, 64, 6),
            nn.ReLU(),
            nn.Conv2d(64, 128, 6),
            nn.ReLU(),
            nn.Conv2d(128, 256, 6),
            nn.ReLU(),
            nn.Conv2d(256, 512, 5),
        ]
        ImageEncoder += [nn.Tanh()]

        self.ImageEncoder = nn.Sequential(*ImageEncoder)

        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, (4, 4, 2), (2, 2, 2)),  # (4, 4, 2)
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, 1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, (2, 2, 2), (2, 2, 2)),  # (8, 8, 4)
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(256, 256, 1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, (2, 2, 2), (2, 2, 2), (0, 0, 1)),  # (16, 16, 6)
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.ConvTranspose3d(128, 1, (2, 2, 1), (2, 2, 1), (1, 1, 0)),  # (16, 16, 6)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.ImageEncoder(x)
        out = out.unsqueeze(-1)
        # print(out.size())  # torch.Size([Batch, 128, 1, 1, 1])
        out = self.layer1(out)
        # print(out.size())  # torch.Size([Batch, 512, 4, 4, 2])
        out = self.layer2(out)
        # print(out.size())  # torch.Size([Batch, 512, 4, 4, 2])
        out = self.layer3(out)
        # print(out.size())  # torch.Size([Batch, 256, 8, 8, 4])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([Batch, 256, 8, 8, 4])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([Batch, 128, 16, 16, 6])
        out = self.layer6(out)
        # print(out.size())  # torch.Size([Batch, 128, 16, 16, 6])
        out = self.layer7(out)
        # print(out.size())  # torch.Size([Batch, 1, 30, 30, 6])
        return out


# 3D->3D for ferro Unet with interpolate upsampling
class UnetGen3DFerro1(nn.Module):
    def __init__(self):
        super(UnetGen3DFerro1, self).__init__()
        self.encoder1 = nn.Conv3d(2, 32, 3, 1, 1)
        self.encoder2 = nn.Conv3d(32, 64, 3, 1, 1)
        self.encoder3 = nn.Conv3d(64, 128, 3, 1, 1)
        self.encoder4 = nn.Conv3d(128, 256, 3, 1, 1)
        self.encoder5 = nn.Conv3d(256, 512, 3, 1, 1)

        self.decoder1 = nn.Conv3d(512, 256, 3, 1, 1)
        self.decoder2 = nn.Conv3d(512, 128, 3, 1, 1)
        self.decoder3 = nn.Conv3d(256, 64, 3, 1, 1)
        self.decoder4 = nn.Conv3d(128, 32, 3, 1, 1)
        self.decoder5 = nn.Conv3d(64, 1, 3, 1, 1)

        self.map = nn.Sigmoid()

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), (2, 2, 1), (2, 2, 1), (1, 1, 0)))  # (16, 16, 6)
        t1 = out  # 32
        out = F.relu(F.max_pool3d(self.encoder2(out), (2, 2, 1), (2, 2, 1)))  # (8, 8, 6)
        t2 = out  # 64
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2, (0, 0, 1)))  # (4, 4, 4)
        t3 = out  # 128
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))  # (2, 2, 2)
        t4 = out  # 256
        out = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))  # (1, 1, 1)

        # print(out.shape, t4.shape)
        out = F.relu(F.interpolate(self.decoder1(out), size=(2, 2, 2), mode='trilinear'))  # 256
        out = torch.cat((out, t4), 1)  # 512
        out = F.relu(F.interpolate(self.decoder2(out), size=(4, 4, 4), mode='trilinear'))  # 128
        out = torch.cat((out, t3), 1)  # 256
        out = F.relu(F.interpolate(self.decoder3(out), size=(8, 8, 6), mode='trilinear'))  # 64
        out = torch.cat((out, t2), 1)  # 128
        out = F.relu(F.interpolate(self.decoder4(out), size=(16, 16, 6), mode='trilinear'))  # 32
        out = torch.cat((out, t1), 1)  # 64
        out = self.map(F.interpolate(self.decoder5(out), size=(30, 30, 6), mode='trilinear'))  # 1

        return out


# 3D->3D for ferro Unet with convtranspose3d upsampling
class UnetGen3DFerro2(nn.Module):
    def __init__(self):
        super(UnetGen3DFerro2, self).__init__()
        self.encoder1 = nn.Conv3d(2, 32, 3, 1, 1)
        self.encoder2 = nn.Conv3d(32, 64, 3, 1, 1)
        self.encoder3 = nn.Conv3d(64, 128, 3, 1, 1)
        self.encoder4 = nn.Conv3d(128, 256, 3, 1, 1)
        self.encoder5 = nn.Conv3d(256, 512, 3, 1, 1)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 2, 2),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, 2, 2, (0, 0, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, (2, 2, 1), (2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, (2, 2, 1), (2, 2, 1), (1, 1, 0)),
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), (2, 2, 1), (2, 2, 1), (1, 1, 0)))  # (16, 16, 6)
        t1 = out  # 32
        out = F.relu(F.max_pool3d(self.encoder2(out), (2, 2, 1), (2, 2, 1)))  # (8, 8, 6)
        t2 = out  # 64
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2, (0, 0, 1)))  # (4, 4, 4)
        t3 = out  # 128
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))  # (2, 2, 2)
        t4 = out  # 256
        out = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))  # (1, 1, 1)

        # print(out.shape, t4.shape)
        out = self.decoder1(out)  # c:256
        out = torch.cat((out, t4), 1)  # c:512
        out = self.decoder2(out)  # c:128
        out = torch.cat((out, t3), 1)  # c:256
        out = self.decoder3(out)  # c:64
        out = torch.cat((out, t2), 1)  # c:128
        out = self.decoder4(out)  # c:32
        out = torch.cat((out, t1), 1)  # c:64
        out = self.decoder5(out)  # c:1

        return out


class ResUGen3DFerro(nn.Module):
    def __init__(self, block, num):
        super(ResUGen3DFerro, self).__init__()
        self.block = block
        self.num = num
        # self.resnet_block = self.build_resnet_block(dim, width)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(2, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            self.build_resnet_block(32)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            self.build_resnet_block(64)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            self.build_resnet_block(128)
        )
        self.encoder4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            self.build_resnet_block(256)
        )
        self.encoder5 = nn.Sequential(
            nn.Conv3d(256, 512, 3, 1, 1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            self.build_resnet_block(512)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, 2, 2),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            self.build_resnet_block(256)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, 2, 2),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            self.build_resnet_block(128)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, 2, 2, (0, 0, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            self.build_resnet_block(64)
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, (2, 2, 1), (2, 2, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            self.build_resnet_block(32)
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, (2, 2, 1), (2, 2, 1), (1, 1, 0)),
        )

    def build_resnet_block(self, dim, width=64):
        resnet_block = []
        if self.block == 'Basic':
            for i in range(self.num):
                resnet_block += [BasicBlock(dim)]
        if self.block == 'Bottle':
            for i in range(self.num):
                resnet_block += [BottleNeck(dim, width)]

        return nn.Sequential(*resnet_block)

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), (2, 2, 1), (2, 2, 1), (1, 1, 0)))  # (16, 16, 6)
        t1 = out  # 32
        out = F.relu(F.max_pool3d(self.encoder2(out), (2, 2, 1), (2, 2, 1)))  # (8, 8, 6)
        t2 = out  # 64
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2, (0, 0, 1)))  # (4, 4, 4)
        t3 = out  # 128
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))  # (2, 2, 2)
        t4 = out  # 256
        out = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))  # (1, 1, 1)

        # print(out.shape, t4.shape)
        out = self.decoder1(out)
        out = torch.cat((out, t4), 1)
        out = self.decoder2(out)
        out = torch.cat((out, t3), 1)
        out = self.decoder3(out)
        out = torch.cat((out, t2), 1)
        out = self.decoder4(out)
        out = torch.cat((out, t1), 1)
        out = self.decoder5(out)

        return out


# -----------------------------Generator for channel-------------------------------------------


# 2D->2D for channel Unet with covtranspose upsampling
class UnetGen2DChan(nn.Module):
    def __init__(self):
        super(UnetGen2DChan, self).__init__()
        self.encoder1 = nn.Conv2d(2, 32, 3, 1, 1)
        self.encoder2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.encoder3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.encoder4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.encoder5 = nn.Conv2d(256, 512, 3, 1, 1)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, 2),  # (2, 2)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, 2, 2),  # (4, 4)
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, 2, 2),   # (8, 8)
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, 2, 2),   # (16, 16)
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, 2, 2, 1),  # (30, 30)
            nn.Sigmoid()
        )

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2, 1))  # (16, 16)
        t1 = out  # 32
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))   # (8, 8)
        t2 = out  # 64
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))   # (4, 4)
        t3 = out  # 128
        out = F.relu(F.max_pool2d(self.encoder4(out), 2, 2))   # (2, 2)
        t4 = out  # 256
        out = F.relu(F.max_pool2d(self.encoder5(out), 2, 2))   # (1, 1)

        # print(out.shape, t4.shape)
        out = self.decoder1(out)       # c:256
        out = torch.cat((out, t4), 1)  # c:512
        out = self.decoder2(out)       # c:128
        out = torch.cat((out, t3), 1)  # c:256
        out = self.decoder3(out)       # c:64
        out = torch.cat((out, t2), 1)  # c:128
        out = self.decoder4(out)       # c:32
        out = torch.cat((out, t1), 1)  # c:64
        out = self.decoder5(out)       # c:1

        return out


# 3D->3D Unet with convtranspose3d upsampling
class UnetGen3DChan(nn.Module):
    def __init__(self):
        super(UnetGen3DChan, self).__init__()
        self.encoder1 = nn.Conv3d(2, 32, 3, 1, 1)
        self.encoder2 = nn.Conv3d(32, 64, 3, 1, 1)
        self.encoder3 = nn.Conv3d(64, 128, 3, 1, 1)
        self.encoder4 = nn.Conv3d(128, 256, 3, 1, 1)
        self.encoder5 = nn.Conv3d(256, 512, 3, 1, 1)

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, (2, 2, 1), (2, 2, 1)),          # (2, 2, 1)
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, (2, 2, 1), (2, 2, 1)),          # (4, 4, 1)
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(256, 64, (2, 2, 1), (2, 2, 1)),           # (8, 8, 1)
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, (2, 2, 2), (2, 2, 2)),           # (16, 16, 2)
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, (2, 2, 1), (2, 2, 1), (1, 1, 0)),  # (30, 30, 2)
        )

    def forward(self, x):
        out = F.relu(F.max_pool3d(self.encoder1(x), (2, 2, 1), (2, 2, 1), (1, 1, 0)))          # (16, 16, 2)
        t1 = out  # 32
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))                                   # (8, 8, 1)
        t2 = out  # 64
        out = F.relu(F.max_pool3d(self.encoder3(out), (2, 2, 1), (2, 2, 1)))                   # (4, 4, 1)
        t3 = out  # 128
        out = F.relu(F.max_pool3d(self.encoder4(out), (2, 2, 1), (2, 2, 1)))                   # (2, 2, 1)
        t4 = out  # 256
        out = F.relu(F.max_pool3d(self.encoder5(out), (2, 2, 1), (2, 2, 1)))                   # (1, 1, 1)

        # print(out.shape, t4.shape)
        out = self.decoder1(out)       # c:256
        out = torch.cat((out, t4), 1)  # c:512
        out = self.decoder2(out)       # c:128
        out = torch.cat((out, t3), 1)  # c:256
        out = self.decoder3(out)       # c:64
        out = torch.cat((out, t2), 1)  # c:128
        out = self.decoder4(out)       # c:32
        out = torch.cat((out, t1), 1)  # c:64
        out = self.decoder5(out)       # c:1

        return out
