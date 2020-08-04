import os 
import torch 
import segmentation_models_pytorch as smp
from config import configs

# define model
model = smp.Unet(
    encoder_name=configs.encoder, 
    encoder_weights=configs.encoder_weights, 
    classes=configs.num_classes, 
    activation=configs.activation)
model.cuda()
model.eval()
# load weights
model_path = "checkpoints/cloud/resnet50-best_iou.pth"
state = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# convert pytorch to onnx
input_img = torch.randn(1,3,configs.input_size,configs.input_size,requires_grad=False,device="cuda")
# export model
save_onnx = "./onnx_models/"
if not os.path.exists(save_onnx):
    os.makedirs(save_onnx)
torch.onnx.export(model,input_img,save_onnx+configs.project+"-"+configs.encoder+".onnx",verbose=True)


