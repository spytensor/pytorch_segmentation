from .boundary_loss import BDLoss,SoftDiceLoss,DC_and_BD_loss,DC_and_HDBinary_loss,DistBinaryDiceLoss,HDDTBinaryLoss
from .dice_loss import DC_and_CE_loss,DC_and_topk_loss,GDiceLoss,GDiceLossV2,SSLoss,IoULoss,TopKLoss,TverskyLoss,FocalTversky_loss
from .dice_loss import AsymLoss,PenaltyGDiceLoss,ExpLog_loss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax
from .ND_Crossentropy import CrossentropyND,WeightedCrossEntropyLoss,WeightedCrossEntropyLossV2,DisPenalizedCE
from torch import nn
losses_seg = {}
losses_seg["BDLoss"] = BDLoss
losses_seg["SoftDiceLoss"] = SoftDiceLoss
losses_seg["DC_and_BD_loss"] = DC_and_BD_loss
losses_seg["DC_and_HDBinary_loss"] = DC_and_HDBinary_loss
losses_seg["DistBinaryDiceLoss"] = DistBinaryDiceLoss
losses_seg["HDDTBinaryLoss"] = HDDTBinaryLoss
losses_seg["DC_and_CE_loss"] = DC_and_CE_loss
losses_seg["DC_and_topk_loss"] = DC_and_topk_loss
losses_seg["GDiceLoss"] = GDiceLoss
losses_seg["GDiceLossV2"] = GDiceLossV2
losses_seg["SSLoss"] = SSLoss
losses_seg["IoULoss"] = IoULoss
losses_seg["TopKLoss"] = TopKLoss
losses_seg["TverskyLoss"] = TverskyLoss
losses_seg["FocalTversky_loss"] = FocalTversky_loss
losses_seg["AsymLoss"] = AsymLoss
losses_seg["PenaltyGDiceLoss"] = PenaltyGDiceLoss
losses_seg["ExpLog_loss"] = ExpLog_loss
losses_seg["FocalLoss"] = FocalLoss
losses_seg["LovaszSoftmax"] = LovaszSoftmax
losses_seg["CrossentropyND"] = CrossentropyND
losses_seg["CrossEntropyLoss"] = nn.CrossEntropyLoss
losses_seg["BCEWithLogitsLoss"] = nn.BCEWithLogitsLoss
losses_seg["WeightedCrossEntropyLoss"] = WeightedCrossEntropyLoss
losses_seg["WeightedCrossEntropyLossV2"] = WeightedCrossEntropyLossV2
losses_seg["DisPenalizedCE"] = DisPenalizedCE

def get_loss_func(name):
    return losses_seg[name]
