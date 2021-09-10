from torch import nn

from .OhemCELoss import OhemCELoss
from .WeightedOhemCELoss import WeightedOhemCELoss
from .detail_loss import DetailAggregateLoss


class Loss(nn.Module):
    def __init__(self, cfg, score_thresh=0.7, ignore_idx=255):
        super(Loss, self).__init__()

        self.cfg = cfg
        self.n_min = cfg.batch_size * cfg.crop_size[0] * cfg.crop_size[1] // 16
        self.loss0 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss1 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss2 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss3 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss4 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss5 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)
        self.loss6 = OhemCELoss(thresh=score_thresh, n_min=self.n_min, ignore_lb=ignore_idx)

    def forward(self, label, d0, d1, d2, d3, d4, d5, d6):
        loss0 = self.loss0(d0, label)
        loss1 = self.loss1(d1, label)
        loss2 = self.loss2(d2, label)
        loss3 = self.loss3(d3, label)
        loss4 = self.loss4(d4, label)
        loss5 = self.loss5(d5, label)
        loss6 = self.loss6(d6, label)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        return loss, [loss0, loss1, loss2, loss3, loss4, loss5, loss6]
