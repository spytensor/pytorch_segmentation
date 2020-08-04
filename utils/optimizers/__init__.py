from .lookahead import *
from .novograd import *
from .over9000 import *
from .radam import *
from .ralamb import *
from .ranger import *
from torch import optim as optim_torch
from config import configs

def get_optimizer(model):
    if configs.optim == "adam":
        return optim_torch.Adam(model.parameters(),
                            configs.lr,
                            betas=(configs.beta1,configs.beta2),
                            weight_decay=configs.wd)
    elif configs.optim == "radam":
        return RAdam(model.parameters(),
                    configs.lr,
                    betas=(configs.beta1,configs.beta2),
                    weight_decay=configs.wd)
    elif configs.optim == "ranger":
        return Ranger(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "over9000":
        return Over9000(model.parameters(),
                        lr = configs.lr,
                        betas=(configs.beta1,configs.beta2),
                        weight_decay=configs.wd)
    elif configs.optim == "ralamb":
        return Ralamb(model.parameters(),
                      lr = configs.lr,
                      betas=(configs.beta1,configs.beta2),
                      weight_decay=configs.wd)
    elif configs.optim == "sgd":
        return optim_torch.SGD(model.parameters(),
                        lr = configs.lr,
                        momentum=configs.mom,
                        weight_decay=configs.wd)
    else:
        print("%s  optimizer will be add later"%configs.optim)