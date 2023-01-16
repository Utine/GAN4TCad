import os
import torch
import numpy as np
import torch.nn.functional as F
from model import *
from random import sample
from datapre import normalize, reshape_cfg, save_samples, clip


def draw_eval_data(percent, solver, save_pth, v):
    pth = './data/raw/' + str(percent) + '_percent'
    configure_list = os.listdir(pth)
    configure_selec = sample(configure_list, 1)
    configure_pth = os.path.join(pth, configure_selec[0])
    configure = np.load(configure_pth + "/polarizatoin_configuration.npy")
    gateV = np.load(configure_pth + "/gate_voltages.npy")

    new_configure = reshape_cfg(configure)
    if solver == 'Ferro':
        ferro = np.load(configure_pth + "/ferro.npy")
        ferroZ = ferro[v][:, :, :, 2]  # (31, 33, 12)
        label_Config, volume_ferroZ = clip(solver, new_configure, ferroZ)  # (30, 30), (30, 30, 6)
        volume = normalize(volume_ferroZ)

    if 'Density' in solver:
        channel = np.load(configure_pth + "/channel.npy")
        density = channel[v][:, :, :, 0]  # (31, 33, 7)
        label_Config, volume_density = clip(solver, new_configure, density)  # (30, 30), (30, 30, 6)
        volume = normalize(volume_density)

    if 'Mobility' in solver:
        channel = np.load(configure_pth + "/channel.npy")
        mobility = channel[v][:, :, :, 1]  # (31, 33, 7)
        label_Config, volume_mobility = clip(solver, new_configure, mobility)  # (30, 30), (30, 30, 6)
        volume = normalize(volume_mobility)

    return gateV[v], new_configure, volume


def start_eval(args):
    if args.solver == "Ferro":
        if args.generator_type == "Normal":
            netG = NormalGen3DFerro()
        elif args.generator_type == "Unet_interpolate":
            netG = UnetGen3DFerro1()
        elif args.generator_type == "Unet_convtranspose":
            netG = UnetGen3DFerro2()
        else:
            netG = ResUGen3DFerro(args.resnet_block_type, args.resnet_block_num)
    elif args.solver == "Current":
        print("not complete yet")
    else:
        if "2D" in args.solver:
            netG = UnetGen2DChan()
        else:
            netG = UnetGen3DChan()
    model_pth = args.model_pth + args.exp + ".pth"
    netG.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))

    val_loss = []
    for idx in range(81):
        voltage, configuration, volume = draw_eval_data(10, args.solver, args.eval_pth, idx)
        configuration = torch.tensor(configuration)
        volume = torch.tensor(volume)
        real = volume

        if args.generator_type == "Normal":
            voltage = torch.stack([torch.ones(configuration.size()[1:]) * i for i in voltage], dim=0)
            input = torch.stack((configuration, voltage), dim=1).type(torch.float32)
        else:
            configuration = torch.stack([configuration for i in range(real.size()[-1])], dim=-1)
            voltage = torch.stack([torch.ones(configuration.size()[1:]) * i for i in voltage], dim=0)
            input = torch.stack((configuration, voltage), dim=1).type(torch.float32)

        pred = netG(input)
        val_loss = val_loss.append(F.l1_loss(pred, volume).detach().numpy())





