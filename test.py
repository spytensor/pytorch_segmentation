import os
import cv2
import torch
import time
import warnings
import random
import segmentation_models_pytorch as smp
from config import configs
from IPython import embed
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations import pytorch as AT
from utils.reader import SegDataset
from utils.utils import iou_metric,dice_metric,AverageMeter
from utils.losses import get_loss_func
from progress.bar import Bar
from PIL import ImageFile 
import albumentations as albu

# set defaul configs
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(configs.seed)

def get_test_augmentation():
    # for test
    test_transform = [
        albu.Resize(height=configs.input_size,width=configs.input_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        AT.ToTensor(),
    ]
    return albu.Compose(test_transform)

def test(test_loader, model, criterion):
    # switch to train mode
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Dice_coeff = AverageMeter()
    Iou = AverageMeter()
    end = time.time()

    bar = Bar('Testing: ', max=len(test_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = model(inputs)
            # compute output
            loss = criterion(outputs, targets)
            # compute iou
            iou_batch = iou_metric(outputs,targets,classes = [str(i) for i in range(configs.num_classes)])
            # compute metric
            dice_batch = dice_metric(outputs,targets)
            # update
            losses.update(loss.item(), inputs.size(0))
            Dice_coeff.update(dice_batch.item(), inputs.size(0))
            Iou.update(iou_batch.mean().item(), inputs.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Dice_coeff: {Dice_coeff: .4f} | Iou: {Iou: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(test_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        Dice_coeff=Dice_coeff.avg,
                        Iou=Iou.avg,
                        )
            bar.next()
        bar.finish()
    print("Test Iou: {}, Test Dice_coeff: {}, Test Loss: {}".format(Iou.avg,Dice_coeff.avg,losses.avg))
if __name__ == "__main__":
    #make dataset
    filenames = glob(configs.test_folder+"masks/*")
    filenames = [os.path.basename(i) for i in filenames]
    test_files = filenames
    test_dataset = SegDataset(test_files,phase="test",transforms = get_test_augmentation())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=configs.bs, shuffle=False, num_workers=configs.workers)
    loss_func = get_loss_func(configs.loss_func)
    criterion = loss_func().cuda()

    # make model
    model = smp.Unet(
        encoder_name=configs.encoder, 
        encoder_weights=configs.encoder_weights, 
        classes=configs.num_classes, 
        activation=configs.activation)
    model.cuda()
    model.eval()
    # model path used to test
    model_path = "checkpoints/cloud/resnet50-best_iou.pth"
    state = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    # do eval
    test(test_loader,model,criterion)
