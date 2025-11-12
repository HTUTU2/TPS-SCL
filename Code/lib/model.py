
import torch.nn as nn
import torch
from Code.lib.Swin import SwinTransformer
from Code.lib.cross import VSSBlock_fuse, VSSBlock_Correlation
from Code.lib.local_Vmamba import VSSBlock
from Code.lib.align import CascadedSpatialFeatureAlignmentNetwork
def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)

def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )
#model

class ALSOD(nn.Module):
    def __init__(self):
        super(ALSOD, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.t_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.fusion = VSSBlock_fuse(hidden_dim=1024)

        self.convAtt4 = conv3x3_bn_relu(1024*2, 1024)
        self.convAtt3 = conv3x3_bn_relu(512*2, 512)
        self.convAtt2 = conv3x3_bn_relu(256*2, 256)
        self.conv1024 = conv3x3_bn_relu(1024, 512)
        self.conv512 = conv3x3_bn_relu(512, 256)
        self.conv256 = conv3x3_bn_relu(256, 128)
        self.conv128 = conv3x3_bn_relu(128, 64)
        self.conv64 = conv3x3(64, 1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)

        self.conv_sem3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv_sem2 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)
        self.VSSBlock4R = VSSBlock(1024, directions=[  'w2', 'w2_flip'], use_tse=True)
        self.VSSBlock4T = VSSBlock(1024,  directions=[  'w7', 'w7_flip'], use_tse=True)
        self.VSSBlock3R = VSSBlock(512,  directions=[  'w2', 'w2_flip'], use_tse=False)
        self.VSSBlock3T = VSSBlock(512,  directions=[  'w7', 'w7_flip'], use_tse=False)
        self.VSSBlock2R = VSSBlock(256,  directions=[ 'w2', 'w2_flip'], use_tse=False)
        self.VSSBlock2T = VSSBlock(256, directions=[  'w7', 'w7_flip'], use_tse=False)

        self.vss_fusion4 = VSSBlock_Correlation(1024, use_tse=True)
        self.vss_fusion3 = VSSBlock_Correlation(512, use_tse=False)
        self.vss_fusion2 = VSSBlock_Correlation(256, use_tse=False)

        in_ch_list = [  256, 512, 1024]
        self.cascaded_alignment = CascadedSpatialFeatureAlignmentNetwork(in_ch_list, grid_size=3)

    def forward(self, rgb, t):
        fr = self.rgb_swin(rgb)#[0-3]
        ft = self.t_swin(t)

        semantic = self.fusion(fr[3], ft[3])
        frr4 = fr[3] * semantic
        ftt4 = ft[3] * semantic
        frr3 = fr[2] * self.conv_sem3(self.up2(semantic))
        ftt3 = ft[2] * self.conv_sem3(self.up2(semantic))
        frr2 = fr[1] * self.conv_sem2(self.up4(semantic))
        ftt2 = ft[1] * self.conv_sem2(self.up4(semantic))

        E_frr4 = self.VSSBlock4R(frr4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        E_ftt4 = self.VSSBlock4T(ftt4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        E_frr3 = self.VSSBlock3R(frr3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        E_ftt3 = self.VSSBlock3T(ftt3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        E_frr2 = self.VSSBlock2R(frr2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        E_ftt2 = self.VSSBlock2T(ftt2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        features_t = [ E_ftt2, E_ftt3, E_ftt4]
        features_rgb = [E_frr2, E_frr3, E_frr4]
        aligned_t = self.cascaded_alignment(features_t, features_rgb)

        rgb4 = self.vss_fusion4( E_frr4, aligned_t[2])
        t4 = self.vss_fusion4( aligned_t[2], E_frr4)
        rgb3 = self.vss_fusion3( E_frr3, aligned_t[1])
        t3 = self.vss_fusion3( aligned_t[1], E_frr3)
        rgb2 = self.vss_fusion2( E_frr2, aligned_t[0])
        t2 = self.vss_fusion2( aligned_t[0], E_frr2)

        r1 = fr[0] + ft[0]

        r4 = self.convAtt4(torch.cat((rgb4, t4), dim=1))
        r3 = self.convAtt3(torch.cat((rgb3, t3), dim=1))
        r2 = self.convAtt2(torch.cat((rgb2, t2), dim=1))
        r4 = self.conv1024(self.up2(r4))
        r3 = self.conv512(self.up2(r3 + r4))
        r2 = self.conv256(self.up2(r2 + r3))
        r1 = self.conv128(r1 + r2)
        out = self.up4(r1)
        out = self.conv64(out)
        return out

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model,weights_only=False)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.t_swin.load_state_dict(torch.load(pre_model,weights_only=False)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")
