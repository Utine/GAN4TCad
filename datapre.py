import os
import numpy as np
import matplotlib.pyplot as plt
from random import sample
# from mayavi import mlab


# process coordinates xy
ferro_x = np.load("./data/xy/ferro_x.npy")
ferro_y = np.load("./data/xy/ferro_y.npy")
ferro_x = (ferro_x - min(ferro_x)) / 0.0025
ferro_y = (ferro_y - min(ferro_y)) / 0.0025


def data_preprocess(args):
    percent_list = os.listdir(args.raw_pth)
    percent_selects = randsamp_by_percent(percent_list, args.percent_samp)
    tag = 0
    for i in range(len(percent_selects)):
        percent_pth = os.path.join(args.raw_pth, percent_selects[i])
        configure_list = os.listdir(percent_pth)
        configure_selects = randsamp_by_percent(configure_list, args.configure_samp)
        for j in range(len(configure_selects)):
            configure_pth = os.path.join(percent_pth, configure_selects[j])
            configure = np.load(configure_pth + "/polarizatoin_configuration.npy")
            gateV = np.load(configure_pth + "/gate_voltages.npy")

            new_configure = reshape_cfg(configure)  # (10, 10) -> (31, 33)

            if args.solver == 'Ferro':
                ferro = np.load(configure_pth + "/ferro.npy")
                for k in range(np.size(gateV)):
                    label_V = gateV[k]
                    ferroZ = ferro[k][:, :, :, 2]  # (31, 33, 12)
                    label_Config, volume_ferroZ = clip(args.solver, new_configure, ferroZ)  # (30, 30), (30, 30, 6)
                    if args.normalize:
                        volume_ferroZ = normalize(volume_ferroZ)
                    # mlab.figure()
                    # mlab.points3d(volume_ferroZ)
                    # mlab.show()
                    save_samples(args.pro_pth + args.exp, tag, label_V, label_Config, volume_ferroZ)
                    tag += 1
                    print(tag)

            if 'Density' in args.solver:
                channel = np.load(configure_pth + "/channel.npy")
                for k in range(np.size(gateV)):
                    label_V = gateV[k]
                    density = channel[k][:, :, :, 0]  # (31, 33, 7)
                    label_Config, volume_density = clip(args.solver, new_configure, density)  # (30, 30), (30, 30, 6)
                    if args.normalize:
                        volume_density = normalize(volume_density)
                    # mlab.figure()
                    # mlab.points3d(volume_density)
                    # mlab.show()
                    save_samples(args.pro_pth + args.exp, tag, label_V, label_Config, volume_density)
                    tag += 1
                    print(tag)

            if 'Mobility' in args.solver:
                channel = np.load(configure_pth + "/channel.npy")
                for k in range(np.size(gateV)):
                    label_V = gateV[k]
                    mobility = channel[k][:, :, :, 1]  # (31, 33, 7)
                    label_Config, volume_mobility = clip(args.solver, new_configure, mobility)  # (30, 30), (30, 30, 6)
                    if args.normalize:
                        volume_mobility = normalize(volume_mobility)
                    # mlab.figure()
                    # mlab.points3d(volume_mobility)
                    # mlab.show()
                    save_samples(args.pro_pth + args.exp, tag, label_V, label_Config, volume_mobility)
                    tag += 1
                    print(tag)


def randsamp_by_percent(dir_list, percent):
    num = round(len(dir_list) * percent)
    select = sample(dir_list, num)
    return select


def reshape_cfg(configure):
    new_configure = np.zeros((np.size(ferro_x), np.size(ferro_y)))
    unit = 4000
    for i in range(np.shape(new_configure)[0]):
        for j in range(np.shape(new_configure)[1]):
            x = ferro_x[i] // unit
            y = ferro_y[j] // unit
            if ferro_x[i] % unit == 0 and ferro_x[i] != 0:
                x = x - 1
            if ferro_y[j] % unit == 0 and ferro_y[j] != 0:
                y = y - 1
            new_configure[i, j] = configure[int(x), int(y)]
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(configure)
    # plt.subplot(1, 2, 2)
    # plt.imshow(new_configure)
    # plt.show()

    return new_configure


def clip(solver, config, volume):
    if solver == "Ferro":
        volume = volume[:30, :30, :6]  # for ferro, (30, 30, 6)
    elif solver == "Current":
        print("current not complete yet")
    else:
        if "2D" in solver:
            volume = volume[:30, :30, 6]  # for channel, only use layer 6
        else:
            volume = volume[:30, :30, :]
            volume = np.delete(volume, [0, 1, 2, 3, 5], 2)  # for channel, only use layer 6 and layer 4 (30, 30, 2)
    config = config[:30, :30]
    return config, volume


def normalize(v):
    v = 2 * (v - v.min()) / (v.max() - v.min()) - 1  # normalize to [-1, 1]
    return v


def save_samples(pth, tag, label_V, label_Config, volume):
    dir_pth = os.path.join(pth, str(tag))
    if not os.path.exists(dir_pth):
        os.makedirs(dir_pth)
    np.save(dir_pth + "/label_V.npy", label_V)
    np.save(dir_pth + "/label_Config.npy", label_Config)
    np.save(dir_pth + "/volume.npy", volume)
