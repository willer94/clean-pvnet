import torch.nn as nn
from lib.utils import net_utils
import torch


def spherical_exp(a):
    """
    a: Bx...xn (Bxn) or (BxHxWxn) or (BxHxWxPxn) tensors
    return as a
    """
    a_exp = a.exp()
    return a_exp / torch.sum(a_exp**2, dim=-1, keepdim=True)


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
    # pr.view(b, h, w, -1)
    # pr = pr.permute(0, 3, 1, 2)
    # gt_abs = gt_abs.permute(0, 2, 3, 1).contiguous()
    gt_abs = gt_abs.view(-1, 2)
    
    return torch.sum(torch.abs(pr[:, 0]*gt_abs[:, 1] - pr[:, 1]*gt_abs[:, 0]))
    # return torch.nn.functional.mse_loss(pr, gt, reduction='sum')

def spherical_sign_loss(pr, gt):
    """
    pr: tensors Bx(PointNum*4)xHxW
    gt: tensors Bx(PointNum)xHxW 0,1,2,3
    """
    pr = pr.permute(0, 2, 3, 1).contiguous()
    b, h, w, vn_4 = pr.shape
    pr = pr.view(b, h, w, vn_4 // 4, 4)
    pr1, pr2 = pr[..., :2], pr[..., 2:]
    pr1, pr2 = torch.softmax(pr1, dim=-1), torch.softmax(pr2, dim=-1)
    gt1, gt2 = gt[..., 0], gt[..., 1]
    l1 = torch.nn.functional.cross_entropy(pr1.view(-1, 2), gt1.flatten(), reduce='sum')
    l2 = torch.nn.functional.cross_entropy(pr2.view(-1, 2), gt2.flatten(), reduce='sum')
    return l1 + l2
    
    # pr = pr.permute(0, 2, 3, 1).contiguous()
    # b, h, w, vn_4 = pr.shape
    # pr = pr.view(b, h, w, vn_4 // 4, 4)
    # pr = pr.view(-1, 4)
    # pr = torch.softmax(pr, -1)
    # gt = gt.permute(0, 2, 3, 1).contiguous()
    # # gt = gt.view(-1, vn_4//4)
    # gt = gt.flatten()
    # return torch.nn.functional.cross_entropy(pr, gt, reduction='sum')


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
            vote_abs_loss = self.vote_abs_crit(output['vertex_abs'] * weight, None, batch['vertex_abs'])
            vote_abs_loss = vote_abs_loss / weight.sum() / batch['vertex_abs'].size(1) * 10
            scalar_stats.update({'vote_abs_loss': vote_abs_loss})
            loss += vote_abs_loss

            mask = batch['mask'].long()
            sign_loss = self.vote_sign_crit(output['vertex_sign'] * weight, batch['vertex_sign'])
            sign_loss = sign_loss / weight.sum() / batch['vertex_abs'].size(1)
            scalar_stats.update({'sign_loss': sign_loss})
            loss += sign_loss

            seg_loss = self.seg_crit(output['seg'], mask)
            scalar_stats.update({'seg_loss': seg_loss})
            loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
