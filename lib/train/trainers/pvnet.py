import torch.nn as nn
from lib.utils import net_utils
import torch


def spherical_exp(a):
    """
    a: Bx...xn (Bxn) or (BxHxWxn) or (BxHxWxPxn) tensors
    return as a
    """
    a_exp = a.exp()
    return a_exp / torch.sqrt(torch.sum(a_exp**2, dim=-1, keepdim=True))


def spherical_abs_loss(pr, gt_sign, gt_abs):
    """
    pr: tensors Bx(PointNum*2)xHxW
    gt_sign: tensors Bx(PointNum*2)xHxW
    gt_abs: tensors Bx(PointNum)xHxW
    """
    pr = pr.permute(0, 2, 3, 1).contiguous()
    b, h, w, vn_2 = pr.shape
    pr = pr.view(-1, 2)
    pr = spherical_exp(pr)
    gt_abs = gt_abs.permute(0, 2, 3, 1).contiguous()
    gt_abs = gt_abs.view(-1, 2)
    
    return torch.sum(torch.abs(pr[:, 0]*gt_abs[:, 1] - pr[:, 1]*gt_abs[:, 0]))
    # return torch.nn.functional.mse_loss(pr, gt, reduction='sum')

def spherical_sign_loss(pr, gt):
    """
    pr: tensors Bx(PointNum*2)xHxW
    gt: tensors Bx(PointNum)xHxW 0,1,2,3
    """
    # return torch.nn.functional.binary_cross_entropy_with_logits(pr, gt.float(), reduction='sum')
    # return torch.nn.functional.smooth_l1_loss(pr, gt.float(), reduction='sum')
    pr = pr.permute(0, 2, 3, 1).contiguous()
    gt = gt.permute(0, 2, 3, 1).contiguous()
    b, h, w, vn_2 = pr.shape
    pr = pr.view(b, h, w, vn_2 // 2, 2)
    pr = pr.view(-1, 2)
    gt = gt.flatten()
    l = torch.nn.functional.cross_entropy(pr, gt, reduction='sum')

    with torch.no_grad():
        acc_num = int(gt.eq(torch.argmax(pr, -1)).sum())
    return l, acc_num


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.spherical_used = net.spherical_used
        self.seg_crit = nn.CrossEntropyLoss()

        if not self.spherical_used:
            self.vote_crit = torch.nn.functional.smooth_l1_loss
        else:
            self.vote_abs_crit = spherical_abs_loss
            self.vote_sign_crit = spherical_sign_loss
            # self.vote_sign_crit = nn.BCEWithLogitsLoss(reduction='sum')
            # self.vote_sign_crit = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        if not self.spherical_used:
            weight = batch['mask'][:, None].float()
            vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
            vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
            scalar_stats.update({'vote_loss': vote_loss})
            loss += vote_loss

            mask = batch['mask'].long()
            seg_loss = self.seg_crit(output['seg'], mask)
            scalar_stats.update({'seg_loss': seg_loss})
            loss += seg_loss
        else:
            weight = batch['mask'][:, None].float()
            vote_abs_loss = self.vote_abs_crit(output['vertex_abs'] * weight, None, batch['vertex_abs']*weight)
            vote_abs_loss = vote_abs_loss / weight.sum() / batch['vertex_abs'].size(1) * 10
            scalar_stats.update({'vote_abs_loss': vote_abs_loss})
            loss += vote_abs_loss

            sign_lossX, x_acc = self.vote_sign_crit(output['vertex_signX'] * weight, batch['vertex_signX']*weight.long())
            sign_lossY, y_acc = self.vote_sign_crit(output['vertex_signY'] * weight, batch['vertex_signY']*weight.long())
            sign_loss = sign_lossX + sign_lossY
            sign_loss = sign_loss / weight.sum() / batch['vertex_abs'].size(1)
            x_acc = (x_acc - (weight.numel() - weight.sum())*9) / weight.sum() / 9
            y_acc = (y_acc - (weight.numel() - weight.sum())*9) / weight.sum() / 9
            scalar_stats.update({'sign_loss': sign_loss})
            scalar_stats.update({'x_acc': x_acc})
            scalar_stats.update({'y_acc': y_acc})
            loss += sign_loss

            mask = batch['mask'].long()
            seg_loss = self.seg_crit(output['seg'], mask)
            scalar_stats.update({'seg_loss': seg_loss})
            loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
