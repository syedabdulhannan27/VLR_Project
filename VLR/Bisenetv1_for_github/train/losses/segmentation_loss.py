from .ohem_ce_loss import OhemCELoss
# import torch
import torch.nn as nn


class SegmentationLoss_nn(nn.Module):
    def __init__(self, conf: dict()):
        super(SegmentationLoss_nn, self).__init__()
        self.conf = conf
        self.LossP = OhemCELoss(self.conf)
        self.Loss2 = OhemCELoss(self.conf)
        self.Loss3 = OhemCELoss(self.conf)

    def compute_loss(self, pred_dict):
        out = pred_dict['out']
        out16, out32 = pred_dict['out16'], pred_dict['out32']
        lb = pred_dict['lb']

        lossp = self.LossP(out, lb)
        loss2 = self.Loss2(out16, lb)
        loss3 = self.Loss3(out32, lb)
        loss = lossp + loss2 + loss3
        return loss

    def backward(self, loss):
        loss.backward()
