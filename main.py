import os 
import cv2 
import torch 
import time 
import random
import warnings
import torchvision as tv
import albumentations as albu
import numpy as np 
from config import configs
from PIL import ImageFile
from glob import glob 
from utils.reader import SegDataset
from utils.losses import *
from utils.optimizers import get_optimizer
from utils.utils import AverageMeter,get_lr,iou_metric,dice_metric,save_checkpoint
from utils.logger import Logger
from utils.metrics import Evaluator
from utils.warmup import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from albumentations import pytorch as AT
import segmentation_models_pytorch as smp
from progress.bar import Bar
from tensorboardX import SummaryWriter

# set defaul configs
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
evaluator = Evaluator(configs.num_classes)
# set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(configs.seed)

# make dir for use
def makdir():
    if not os.path.exists(configs.log_dir):
        os.makedirs(configs.log_dir)
    if not os.path.exists(configs.checkpoints):
        os.makedirs(configs.checkpoints)
    if not os.path.exists(configs.pred_mask):
        os.makedirs(configs.pred_mask)
makdir()

best_iou = 0   
best_dice = 0  
# augumentations
def get_training_augmentation():
    # for train
    train_transform = [
        albu.Resize(height=configs.input_size,width=configs.input_size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        AT.ToTensor(),
    ]
    return albu.Compose(train_transform)
def stong_aug():
    # strong aug for  train
    train_transform = [
        albu.Resize(height=configs.input_size,width=configs.input_size),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.OneOf([
            albu.CenterCrop(p=0.5,height=configs.input_size,width=configs.input_size),
            albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            albu.GridDistortion(p=0.5),
            albu.OpticalDistortion(p=1, distort_limit=1, shift_limit=0.5),
        ],p=0.8),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        AT.ToTensor(),
    ]
    return albu.Compose(train_transform)
def get_valid_augmentation():
    # for valid
    valid_transform = [
        albu.Resize(height=configs.input_size,width=configs.input_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        AT.ToTensor(),
    ]
    return albu.Compose(valid_transform)
def main():
    global best_iou
    global best_dice
    # model 
    model = smp.Unet(
    encoder_name=configs.encoder, 
    encoder_weights=configs.encoder_weights, 
    classes=configs.num_classes, 
    activation=configs.activation)
    if len(configs.gpu_id)>1:
        model = nn.DataParallel(model)
    model.cuda()
    # get files
    filenames = glob(configs.dataset+"masks/*")
    filenames = [os.path.basename(i) for i in filenames]
    # random split dataset into train and val
    train_files, val_files = train_test_split(filenames, test_size=0.2)
    # define different aug
    if configs.use_strong_aug:
        transform_train = stong_aug()
    else:
         transform_train = get_training_augmentation()
    transform_valid = get_valid_augmentation()
    # make data loader for train and val
    train_dataset = SegDataset(train_files,phase="train",transforms = transform_train)
    valid_dataset = SegDataset(val_files,phase="valid",transforms = transform_valid)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.bs, shuffle=True, num_workers=configs.workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=configs.bs, shuffle=False, num_workers=configs.workers)
    optimizer = get_optimizer(model)
    loss_func = get_loss_func(configs.loss_func)
    criterion = loss_func().cuda()
    # tensorboardX writer
    writer = SummaryWriter(configs.log_dir)
    # set lr scheduler method
    if configs.lr_scheduler == "step":
        scheduler_default = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    elif configs.lr_scheduler == "on_loss":
        scheduler_default = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=False)
    elif configs.lr_scheduler == "on_iou":
        scheduler_default = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=False)
    elif configs.lr_scheduler == "on_dice":
        scheduler_default = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5, verbose=False)
    elif configs.lr_scheduler == "cosine":
        scheduler_default = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, configs.epochs-configs.warmup_epo)
    else:
        scheduler_default = torch.optim.lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
    # scheduler with warmup
    if configs.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=configs.warmup_factor, total_epoch=configs.warmup_epo, after_scheduler=scheduler_default)
    else:
        scheduler = scheduler_default
    for epoch in range(configs.epochs):
        print('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, configs.epochs, optimizer.param_groups[0]['lr']))
        train_loss,train_dice,train_iou = train(train_loader,model,criterion,optimizer,epoch,writer)
        valid_loss,valid_dice,valid_iou = eval(valid_loader,model,criterion,epoch,writer)
        if configs.lr_scheduler == "step" or configs.lr_scheduler == "cosine" or configs.warmup:
            scheduler.step(epoch)
        elif configs.lr_scheduler == "on_iou":
            scheduler.step(valid_iou)
        elif configs.lr_scheduler == "on_dice":
            scheduler.step(valid_dice)
        elif configs.lr_scheduler == "on_loss":
            scheduler.step(valid_loss)
        # save model
        is_best_iou = valid_iou > best_iou
        is_best_dice = valid_dice > best_dice
        best_iou = max(valid_iou, best_iou)
        best_dice = max(valid_dice,best_dice)   
        print("Best {}: {} ,Best Dice: {}".format(configs.metric,best_iou,best_dice))
        save_checkpoint({
            'state_dict': model.state_dict(),
        },is_best_iou,is_best_dice)

def train(train_loader, model, criterion, optimizer, epoch,writer):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Dice_coeff = AverageMeter()
    Iou = AverageMeter()
    end = time.time()
    evaluator.reset()
    bar = Bar('Training: ', max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
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
        target = targets.cpu().numpy() 
        pred = outputs.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)                                                                                                                              
        target = np.argmax(target, axis=1)                                                                                                                          
        evaluator.add_batch(target, pred)      
        if configs.metric == "mIoU":
            iou_value = evaluator.Mean_Intersection_over_Union()
        else:
            iou_value = evaluator.Frequency_Weighted_Intersection_over_Union() 
        Iou.update(iou_value, inputs.size(0)) 

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # clip gradient
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Dice_coeff: {Dice_coeff: .4f} | {metric}: {Iou: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    Dice_coeff=Dice_coeff.avg,
                    metric=configs.metric[0],
                    Iou=Iou.avg,
                    )
        writer.add_scalar("Train-Loss",losses.avg,epoch)
        writer.add_scalar("Train-%s"%configs.metric,Iou.avg,epoch)
        writer.add_scalar("Train-Dice",Dice_coeff.avg,epoch)

        bar.next()
    bar.finish()
    return (losses.avg, Dice_coeff.avg, Iou.avg)

def eval(valid_loader, model, criterion, epoch,writer):
    # switch to train mode
    model.eval()
    global best_dice
    global best_iou
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Dice_coeff = AverageMeter()
    Iou = AverageMeter()
    end = time.time()
    evaluator.reset()
    bar = Bar('Validing: ', max=len(valid_loader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valid_loader):
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
            target = targets.cpu().numpy()                      
            pred = outputs.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)                                                                                                                              
            target = np.argmax(target, axis=1)                                                                                                                          
            evaluator.add_batch(target, pred)      
            if configs.metric == "mIoU":
                iou_value = evaluator.Mean_Intersection_over_Union()
            else:
                iou_value = evaluator.Frequency_Weighted_Intersection_over_Union() 
            Iou.update(iou_value, inputs.size(0)) 
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Dice_coeff: {Dice_coeff: .4f} | {metric}: {Iou: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valid_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        Dice_coeff=Dice_coeff.avg,
                        metric=configs.metric[0],
                        Iou=Iou.avg,
                        )
            bar.next()
            writer.add_scalar("Valid-Loss",losses.avg,epoch)
            writer.add_scalar("Valid-%s"%configs.metric,Iou.avg,epoch)
            writer.add_scalar("Valid-Dice",Dice_coeff.avg,epoch)
        bar.finish()
    return (losses.avg, Dice_coeff.avg, Iou.avg)

if __name__ == "__main__":
    main()

