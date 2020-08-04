import os
import cv2
import torch
import segmentation_models_pytorch as smp
from config import configs
from IPython import embed
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from albumentations import pytorch as AT
import albumentations as albu

def get_valid_augmentation():
    # for valid
    valid_transform = [
        albu.Resize(height=configs.input_size,width=configs.input_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
        AT.ToTensor(),
    ]
    return albu.Compose(valid_transform)

model = smp.Unet(
    encoder_name=configs.encoder, 
    encoder_weights=configs.encoder_weights, 
    classes=configs.num_classes, 
    activation=configs.activation)
model.cuda()
model.eval()
state = torch.load("checkpoints/cloud/resnet50-best_dice.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

images = glob("/data/zcj/dataset/detection/cloud/dataset/test/images/*")
#images = glob("./webank/images/*")
for filename in tqdm(images[:20]):
    fname = os.path.basename(filename)
    image = cv2.imread(filename)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = get_valid_augmentation()(image=img)
    img = augmented['image'].unsqueeze(0)
    preds_batch = model(img.cuda())
    for preds in preds_batch:
        pred_mask = np.argmax(preds.detach().cpu().numpy(),0).astype(np.uint8)
        #cv2.imwrite(configs.pred_mask+"pred_"+fname,pred_mask*255)
        plt.figure(figsize=(18,18))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.subplot(1,2,2)
        plt.imshow(pred_mask)
        plt.savefig(configs.pred_mask+"pred_"+fname)
        plt.close()

