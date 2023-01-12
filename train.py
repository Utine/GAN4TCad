import os
import torch
import numpy as np
import wandb
from torch import optim
from torch import nn
from torch.utils.data import random_split, Dataset, DataLoader
from model import *
# from mayavi import mlab


class TCadData(Dataset):

    def __init__(self, args):
        self.pth = args.pro_pth + args.exp

    def __getitem__(self, idx):
        dir_pth = os.path.join(self.pth, str(idx))
        label_V = np.load(dir_pth + "/label_V.npy")
        label_Config = np.load(dir_pth + "/label_Config.npy")
        volume = np.load(dir_pth + "/volume.npy")

        return {'label_V': label_V, 'label_Config': label_Config, 'result': volume}

    def __len__(self):
        return len(os.listdir(self.pth))


class TCad3DBuild:
    def __init__(self, args):
        self.args = args
        wandb.init(project="GAN4TCAD", name=self.args.exp, config=self.args)
        # model define
        if self.args.solver == "Ferro":
            self.netD = Discrim3DFerro()
            if self.args.generator_type == "Normal":
                self.netG = NormalGen3DFerro()
            elif self.args.generator_type == "Unet_interpolate":
                self.netG = UnetGen3DFerro1()
            elif self.args.generator_type == "Unet_convtranspose":
                self.netG = UnetGen3DFerro2()
            else:
                self.netG = ResUGen3DFerro(self.args.resnet_block_type, self.args.resnet_block_num)
        elif self.args.solver == "Current":
            print("not complete yet")
        else:
            if "2D" in self.args.solver:
                self.netD = Discrim2DChan()
                self.netG = UnetGen2DChan()
            else:
                self.netD = Discrim3DChan()
                self.netG = UnetGen3DChan()

        if torch.cuda.is_available():
            self.device = self.args.device

        self.netD.to(self.device)
        self.netG.to(self.device)

        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=args.d_lr)
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=args.g_lr)
        self.criterionGAN = nn.BCEWithLogitsLoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)

        # load dataset
        dset = TCadData(self.args)
        tr_dset, va_dset = random_split(dset, [int(len(dset) * 0.9), len(dset) - int(len(dset) * 0.9)])
        self.tr_loader = DataLoader(tr_dset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        self.va_loader = DataLoader(va_dset, 1)

        # trick setting
        if self.args.soft_label:
            self.real_labels = torch.randn(self.args.batch_size).uniform_(1, 1.2)
            self.fake_labels = torch.randn(self.args.batch_size).uniform_(0, 0.3)
        else:
            self.real_labels = torch.ones(self.args.batch_size)
            self.fake_labels = torch.zeros(self.args.batch_size)

    # calculate loss for discriminator
    def lossD(self):
        # calculate loss for generated volumes
        pred_fake = self.netD(self.fake.detach())
        loss_fake = self.criterionGAN(pred_fake.squeeze(), self.fake_labels)
        # calculate loss for real volumes
        pred_real = self.netD(self.real.unsqueeze(1))
        loss_real = self.criterionGAN(pred_real.squeeze(), self.real_labels)
        # compose loss for discriminator
        self.loss_D = loss_real + loss_fake
        wandb.log({"netD-FakeLoss": loss_fake, "netD-RealLoss": loss_real, "netD-Loss": self.loss_D})

    # calculate loss for generator
    def lossG(self):
        pred_fake = self.netD(self.fake)
        # 1. netG should fake netD
        loss_GAN = self.criterionGAN(pred_fake.squeeze(), self.real_labels)
        # 2. generated volumes should match the real
        loss_L1 = self.criterionL1(self.real.unsqueeze(1), self.fake)
        # compose loss for generator
        self.loss_G = loss_GAN + loss_L1 * self.args.lamda
        wandb.log({"netG-BCELoss": loss_GAN, "netG-L1Loss": loss_L1, "netG-Loss": self.loss_G})

    # calculate prediction accuracy of discriminator
    def accD(self):
        pred_fake = self.netD(self.fake)
        pred_real = self.netD(self.real.unsqueeze(1))
        real_acc = torch.ge(pred_real.squeeze(), 0.5).float()
        fake_acc = torch.le(pred_fake.squeeze(), 0.5).float()
        self.acc_D = torch.mean(torch.cat((real_acc, fake_acc), 0))
        wandb.log({"netD-ACC": self.acc_D})

    def train(self):
        for epoch in range(self.args.n_epochs):
            for i, dic in enumerate(self.tr_loader):
                configuration = dic['label_Config']
                voltage = dic['label_V']
                self.real = dic['result'].to(self.device)  # (32, 30, 30, 8)

                if self.args.generator_type == "Normal":
                    voltage = torch.stack([torch.ones(configuration.size()[1:]) * i for i in voltage], dim=0)
                    input = torch.stack((configuration, voltage), dim=1).type(torch.float32)
                else:
                    configuration = torch.stack([configuration for i in range(self.real.size()[-1])], dim=-1)
                    voltage = torch.stack([torch.ones(configuration.size()[1:]) * i for i in voltage], dim=0)
                    input = torch.stack((configuration, voltage), dim=1).type(torch.float32)  # (32, 2, 30, 30, 8)
                input = input.to(self.device)
                # mlab.figure('config')
                # mlab.points3d(configuration)
                # mlab.figure('voltage')
                # mlab.points3d(voltage)
                # mlab.show()
                # generate fake volumes
                self.fake = self.netG(input)

                # train the discriminator
                self.lossD()
                self.accD()

                if self.acc_D <= self.args.d_thresh:
                    self.optimizer_D.zero_grad()
                    self.loss_D.backward()
                    self.optimizer_D.step()

                if self.args.sh_lr:
                    try:
                        self.scheduler_D.step()
                    except Exception as e:
                        print("fail lr scheduling", e)

                # train the generator
                self.lossG()
                self.optimizer_D.zero_grad()
                self.optimizer_G.zero_grad()
                self.loss_G.backward()
                self.optimizer_G.step()

            # save infos, show results
            val_loss, percent_min, percent_max, percent_mean, abnormals = self.eval()
            self.output_log = 'Epoch-{} , D_loss : {:.4}, G_loss : {:.4}, D_acu : {:.4}, val_loss : {:.4}, percent_min : {:.4}, percent_max : {:.4}, percent_mean : {:.4}, ' \
                              'abnormals : {}'.format(epoch, self.loss_D, self.loss_G, self.acc_D, val_loss, percent_min, percent_max, percent_mean, abnormals)

            if (epoch + 1) % self.args.save_step == 0:
                self.iteration = epoch
                self.save()

            print(self.output_log)
        model_pth = self.args.model_pth + self.args.exp + ".pth"
        torch.save(self.netG.state_dict(), model_pth)

    def eval(self):
        val_loss = []
        percent_min = []
        percent_max = []
        percent_mean = []
        abnormal_num = []
        for i, dic in enumerate(self.va_loader):
            cf = dic['label_Config']
            v = dic['label_V']
            target = dic['result']  # (32, 30, 30, 8)
            target = target.unsqueeze(1)

            if self.args.generator_type == "Normal":
                v = torch.stack([torch.ones(cf.size()[1:]) * i for i in v], dim=0)
                x = torch.stack((cf, v), dim=1).type(torch.float32)
            else:
                cf = torch.stack([cf for i in range(target.size()[-1])], dim=-1)
                v = torch.stack([torch.ones(cf.size()[1:]) * i for i in v], dim=0)
                x = torch.stack((cf, v), dim=1).type(torch.float32)  # (32, 2, 30, 30, 8)

            pred = self.netG(x).detach()
            val_loss.append(self.criterionL1(target, pred))
            percent = torch.abs((pred - target).div_(target))
            # filter Nan and Inf
            abnormal = np.sum(torch.isnan(percent).numpy() != 0) + np.sum(torch.isinf(percent).numpy() != 0)
            percent = torch.where(torch.isnan(percent), torch.full_like(percent, 0.1), percent)
            percent = torch.where(torch.isinf(percent), torch.full_like(percent, 0.1), percent)
            error_min, error_max, error_mean = percent.min(), percent.max(), percent.mean()

            percent_min.append(error_min)
            percent_max.append(error_max)
            percent_mean.append(error_mean)
            abnormal_num.append(abnormal)

        return sum(val_loss)/len(val_loss), sum(percent_min)/len(percent_min), sum(percent_max)/len(percent_max),\
            sum(percent_mean)/len(percent_min), sum(abnormal_num)/len(abnormal_num)

    def save(self):
        # save volumes
        volume_pth = self.args.result_pth + self.args.exp + "/volumes"
        if not os.path.exists(volume_pth):
            os.makedirs(volume_pth)
        np.save(volume_pth + '/fake{}.npy'.format(str(self.iteration)), self.fake[0][0].detach().numpy())
        np.save(volume_pth + '/real{}.npy'.format(str(self.iteration)), self.real[0].detach().numpy())

        # save training logs
        log_pth = self.args.result_pth + self.args.exp
        if not os.path.exists(log_pth):
            os.makedirs(log_pth)
        with open(log_pth + '/logs.txt', 'a+') as f:
            f.write(self.output_log + '\n')

        # save 3d plots
        # mlab.figure()
        # mlab.points3d(self.fake[0][0])
        # mlab.savefig(volume_pth + '/fake{}.png'.format(str(self.iteration)))
        # mlab.clf()
        # mlab.points3d(self.real[0])
        # mlab.savefig(volume_pth + '/real{}.png'.format(str(self.iteration)))
        # mlab.close(all=True)
