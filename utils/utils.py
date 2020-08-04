import torch 
import numpy as np 
from functools import partial
import os 
from config import configs
import shutil
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def get_activation_fn(activation: str = None):
    """Returns the activation function from ``torch.nn`` by its name."""
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def iou_metric(outputs,targets,classes=None,eps= 1e-7,threshold= None,activation= "Sigmoid"):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        Union[float, List[float]]: IoU (Jaccard) score(s)
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    # ! fix backward compatibility
    if classes is not None:
        # if classes are specified we reduce across all dims except channels
        _sum = partial(torch.sum, dim=[0, 2, 3])
    else:
        _sum = torch.sum

    intersection = _sum(targets * outputs)
    union = _sum(targets) + _sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than IoU == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    iou = (intersection + eps * (union == 0)) / (union - intersection + eps)

    return iou
def dice_metric(outputs,targets,eps = 1e-7,threshold = None,activation = "Sigmoid",):
    """Computes the dice metric.
    Args:
        outputs (list):  a list of predicted elements
        targets (list): a list of elements that are to be predicted
        eps (float): epsilon
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs)
    # this looks a bit awkward but `eps * (union == 0)` term
    # makes sure that if I and U are both 0, than Dice == 1
    # and if U != 0 and I == 0 the eps term in numerator is zeroed out
    # i.e. (0 + eps) / (U - 0 + eps) doesn't happen
    dice = (2 * intersection + eps * (union == 0)) / (union + eps)

    return dice

def save_checkpoint(state,is_best_iou,is_best_dice):
    filename = configs.checkpoints + os.sep + configs.encoder + "-checkpoint.pth"
    torch.save(state, filename)
    if is_best_iou:
        message = filename.replace("-checkpoint.pth","-best_iou.pth")
        shutil.copyfile(filename, message)
    if is_best_dice:
        message = filename.replace("-checkpoint.pth","-best_dice.pth")
        shutil.copyfile(filename, message)
