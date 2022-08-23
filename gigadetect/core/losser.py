#!/usr/bin/python3

"""
# -*- coding: utf-8 -*-

# @Time     : 2020/8/11 13:28
# @File     : losser.py

"""
import torch
from torch import nn

from gigadetect.utils import geometry


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

def build_targets(cfg, p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    number_anchor, number_target = det.na, targets.shape[0]  # number of anchors, targets
    target_cls, target_box, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(number_anchor, device=targets.device).float().view(number_anchor, 1).repeat(1, number_target)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(number_anchor, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if number_target:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < cfg.MODEL.ANCHOR_THRESHOLD # compare
            # j = wh_iou(anchors, t[:, 4:6]) > cfg.MODEL.IOU_THRESHOLD  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        target_box.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        target_cls.append(c)  # class

    return target_cls, target_box, indices, anch


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def compute_loss(cfg, p, targets, model):  # predictions, targets, model
    device = targets.device
    loss_cls, loss_bbox, loss_obj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    target_cls, tbox, indices, anchors = build_targets(cfg, p, targets, model)  # targets

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.MODEL.LOSS.BCE_CLS_POSITIVE_WEIGHT])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([cfg.MODEL.LOSS.BCE_OBJ_POSITIVE_WEIGHT])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    use_focal_loss = cfg.MODEL.LOSS.FOCAL_LOSS.ENABLED
    if use_focal_loss:
        focal_loss_gamma = cfg.MODEL.LOSS.FOCAL_LOSS.GAMMA
        BCEcls, BCEobj = FocalLoss(BCEcls, focal_loss_gamma), FocalLoss(BCEobj, focal_loss_gamma)

    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        target_obj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = geometry.bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
            loss_bbox += (1.0 - giou).mean()  # giou loss

            # Objectness
            target_obj[b, a, gj, gi] = (1.0 - cfg.MODEL.LOSS.GIOU_LOSS_RATIO) + cfg.MODEL.LOSS.GIOU_LOSS_RATIO * giou.detach().clamp(0).type(target_obj.dtype)  # giou ratio

            # Classification
            if cfg.DATASETS.NUM_CLASSES > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), target_cls[i]] = cp
                loss_cls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        loss_obj += BCEobj(pi[..., 4], target_obj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    loss_bbox *= cfg.MODEL.LOSS.GIOU_GAIN * s
    loss_obj *= cfg.MODEL.LOSS.OBJ_GAIN * s * (1.4 if np == 4 else 1.)
    loss_cls *= cfg.MODEL.LOSS.CLS_GAIN * s
    bs = target_obj.shape[0]  # batch size
    loss = loss_bbox + loss_obj + loss_cls
    return loss * bs, torch.cat((loss_bbox, loss_obj, loss_cls, loss)).detach()

