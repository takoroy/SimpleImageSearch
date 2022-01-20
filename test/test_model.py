from model import ArcMargineLoss
import torch


def test_loss():
    loss_func = ArcMargineLoss(100, 256, 0.5, 1.0)
    out = torch.rand(32, 256)
    label = torch.empty(32).long().random_(0, 100)
    loss = loss_func(out, label)
    assert loss.item() > 0
