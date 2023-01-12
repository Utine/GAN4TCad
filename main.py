import yaml
import os
from train import TCad3DBuild
from datapre import data_preprocess
from utils import parser_from_dict, make_params_string, plot_volumes

if __name__ == "__main__":
    f = open("./parameters.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)

    # save experiment config
    log_param = make_params_string(args)
    pth = args.result_pth + args.exp
    if not os.path.exists(pth):
        os.makedirs(pth)
    with open(pth + '/config.txt', 'w+') as f:
        f.write(log_param)

    # preprocess raw data
    data_preprocess(args)

    # start training
    builder = TCad3DBuild(args)
    builder.train()

    # show rebuild results during training
    # plot_volumes('results/Density_Exp2_Unet_convtranspose/volumes', 10)
