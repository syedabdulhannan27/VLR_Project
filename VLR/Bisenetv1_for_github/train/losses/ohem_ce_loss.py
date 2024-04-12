#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

device_num = 0


class OhemCELoss(nn.Module):
    def __init__(self, conf: dict()):    #thresh, n_min, ignore_lb=255, *args, **kwargs
        super(OhemCELoss, self).__init__()
        self.loss_dict = conf['training_dict']
        self.thresh = -torch.log(torch.tensor(self.loss_dict['loss_thresh'],
                                              dtype=torch.float)).cuda(device_num)
        # self.n_min = self.loss_dict['n_img_per_gpu'] * self.loss_dict['cropsize'][0] * self.loss_dict['cropsize'][1]//24
        self.n_min = 16000
        self.ignore_lb = self.loss_dict['ignore_lb']
        self.criteria = nn.CrossEntropyLoss(
            ignore_index=self.loss_dict['ignore_lb'],
            reduction='none'
            )

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        labels = labels.long()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
    
    # def compute_loss(self, pred_dict: dict()):
    #     lb = pred_dict['lb']
    #     out = pred_dict['out']
    #     out16 = pred_dict['out16']
    #     out32 = pred_dict['out32']
    #     lossp = self.forward(out, lb)
    #     loss2 = self.forward(out16, lb)    #Loss2(out16, lb)
    #     loss3 = self.forward(out32, lb)   #Loss3(out32, lb)
    #     loss = lossp + loss2 + loss3
    #     return loss


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
