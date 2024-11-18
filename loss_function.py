import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import monai


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.Loss = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        result = self.Loss(predict, target)
        return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1e-5, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        # predict = torch.sigmoid(predict)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, ignore_index=[10000]):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.b_dice = BinaryDiceLoss()

        self.coff = [0.1, 0.9]

    def forward(self, predict, target):
        num_labels = predict.shape[1]
        predict = torch.softmax(predict, dim=1)
        total_loss = []

        for i in range(num_labels):
            if i not in self.ignore_index:
                target_clone = target.clone().float()
                target_clone[target_clone == i] = 1
                binary_dice_loss = self.b_dice(predict[:, i, :, :].clone(), target_clone)
                total_loss.append(binary_dice_loss * self.coff[i])

        total_loss = torch.stack(total_loss, dim=0)

        return total_loss.mean()


class CE_Dice_Loss(nn.Module):
    def __init__(self):
        super(CE_Dice_Loss, self).__init__()
        self.CE_Loss = nn.CrossEntropyLoss(reduction="mean")
        self.Dice_Loss = DiceLoss()

    def forward(self, predict, target):
        result = self.CE_Loss(predict, target) + self.Dice_Loss(predict, target)
        return result


class BCE_Dice_Loss(nn.Module):
    def __init__(self):
        super(BCE_Dice_Loss, self).__init__()
        self.CE_Loss = nn.BCELoss(reduction="mean")
        self.Dice_Loss = BinaryDiceLoss()

    def forward(self, predict, target):
        result = self.CE_Loss(predict, target.unsqueeze(1)) + self.Dice_Loss(predict, target.unsqueeze(1))
        # result_self = softmax_entropy(predict)
        # result = result + result_self.mean()
        return result


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    return -(x.softmax(dim=1) * x.log_softmax(dim=1)).sum(1)


if __name__ == "__main__":

    fc_proj = nn.Parameter(torch.FloatTensor(2, 3))
    print(fc_proj)
