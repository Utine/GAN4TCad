import torch
import argparse
import numpy as np
# from mayavi import mlab
from collections import OrderedDict


def gen_noise(args):
    if args.noise_dis == "norm":
        noise = torch.Tensor(args.batch_size, args.z_size).normal_(0, 0.33)
    if args.noise_dis == "uni":
        noise = torch.randn(args.batch_size, args.z_size)

    return noise


def set_require_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def make_params_string(args):
    params_list = [("exp", args.exp),
                   ("percent_samp", args.percent_samp),
                   ("configure_samp", args.configure_samp),
                   ("resnetblock", args.resnet_block_type),
                   ("num", args.resnet_block_num),
                   ("generator", args.generator_type),
                   ("bs", args.batch_size),
                   ("g_lr", args.g_lr),
                   ("d_lr", args.d_lr),
                   ("lamda", args.lamda),
                   ("sl", args.soft_label), ]
    params_dict = OrderedDict(((arg, value) for arg, value in params_list))
    str_result = ""
    for i in params_dict.keys():
        str_result = str_result + str(i) + '=' + str(params_dict[i]) + '\n'
    return str_result[:-1]


def parser_from_dict(dic):
    parser = argparse.ArgumentParser()
    for k, v in dic.items():
        parser.add_argument("--" + k, default=v)
    args = parser.parse_args()

    return args


def plot_volumes(pth, epoch):
    fake = np.load(pth+'/fake{}.npy'.format(str(epoch)))
    real = np.load(pth+'/real{}.npy'.format(str(epoch)))
    mlab.figure('fake')
    mlab.points3d(fake)
    mlab.figure('real')
    mlab.points3d(real)
    mlab.show()
    mlab.close()


# def concatenate(img, volume):
#     #  volume(8, 30, 30)
#     #  img(30, 30)
#     cat_v_i = torch.cat(list(torch.split(volume, 1, dim=0)).append(img))
#     return cat_v_i

# plot_volumes('/Users/yutinghu/PycharmProjects/gan4tcad/results/volumes/model=tcad3dbuild_bs=32_g_lr=0.001_d_lr=0.001_lamda=10_sl=True', 37)
