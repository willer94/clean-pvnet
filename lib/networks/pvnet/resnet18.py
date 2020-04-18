from torch import nn
import torch
from torch.nn import functional as F
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg


def spherical_exp(a):
    """
    a: Bx...xn (Bxn) or (BxHxWxn) or (BxHxWxPxn) tensors
    return as a
    """
    a_exp = a.exp()
    return a_exp / torch.sum(a_exp**2, dim=-1, keepdim=True)


class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, spherical_used=False, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim=ver_dim
        self.seg_dim=seg_dim
        self.spherical_used = spherical_used

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1)
        )
        if self.spherical_used:
            self.convsign = nn.Sequential(
                nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
                nn.BatchNorm2d(raw_dim),
                nn.LeakyReLU(0.1,True),
                nn.Conv2d(raw_dim, ver_dim*2, 1, 1) # N * 4, label have 4 dims
            )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2//2, 2)
        mask = torch.argmax(output['seg'], 1)
        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def decode_keypoint_spherical(self, output):
        vertex_abs  = output['vertex_abs'].permute(0, 2, 3, 1)
        vertex_sign = output['vertex_sign'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex_abs.shape
        vertex_abs = vertex_abs.view(b, h, w, vn_2//2, 2)
        vertex_abs = spherical_exp(vertex_abs)
        # pr1, pr2 = pr[..., :2], pr[..., 2:]
        # pr1, pr2 = torch.softmax(pr1, dim=-1), torch.softmax(pr2, dim=-1)
        vertex_sign = vertex_sign.view(b, h, w, vn_2//2, 4)
        vertex_sign1, vertex_sign2 = vertex_sign[..., :2], vertex_sign[..., 2:]
        vertex_sign1, vertex_sign2 = torch.argmax(vertex_sign1, -1), torch.argmax(vertex_sign2, -1)
        vertex_sign = torch.stack((vertex_sign1, vertex_sign2), dim=-1) * 2 - 1
        vertex = vertex_abs * vertex_sign.to(device=vertex_sign.device, dtype=torch.float32)
        mask = torch.argmax(output['seg'], 1)
        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})


    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        fm = torch.cat([fm,x],1)
        x=self.convraw(fm)

        if not self.spherical_used:
            seg_pred=x[:,:self.seg_dim,:,:]
            ver_pred=x[:,self.seg_dim:,:,:]

            ret = {'seg': seg_pred, 'vertex': ver_pred}

            if not self.training:
                with torch.no_grad():
                    self.decode_keypoint(ret)
        else:
            seg_pred = x[:,:self.seg_dim,:,:]
            ver_abs  = x[:,self.seg_dim:,:,:]
            ver_sign = self.convsign(fm)
            ret = {'seg': seg_pred, 'vertex_abs': ver_abs, 'vertex_sign': ver_sign}

            if not self.training:
                with torch.no_grad():
                    self.decode_keypoint_spherical(ret)

        return ret


def get_res_pvnet(ver_dim, seg_dim, spherical_used=False):
    model = Resnet18(ver_dim, seg_dim, spherical_used)
    return model

