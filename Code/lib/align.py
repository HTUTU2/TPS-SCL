import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFeatureAlignmentNetwork(nn.Module):
    def __init__(self, in_ch, grid_size=5):
        super(SpatialFeatureAlignmentNetwork, self).__init__()
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch * 2, self.num_points * 2)
        initial_points = self._build_initial_control_points()
        self.register_buffer('initial_control_points', initial_points)  # 保持原名

    def _build_initial_control_points(self):
        control_points = torch.linspace(-1, 1, steps=self.grid_size)
        p, q = torch.meshgrid(control_points, control_points, indexing='ij')
        initial_control_points = torch.stack([p.flatten(), q.flatten()], dim=1)  # (num_points, 2)
        return initial_control_points

    def forward(self, x, prev_control_points=None):

        t = x[0]
        rgb = x[1]
        in_feat = torch.cat([t, rgb], dim=1)  # (batch_size, in_ch * 2, H, W)
        batch_size = in_feat.size(0)
        features = self.pool(in_feat).view(batch_size, -1)  # (batch_size, in_ch)
        delta_control_points = self.fc(features)  # (batch_size, num_points * 2)
        delta_control_points = delta_control_points.view(batch_size, self.num_points, 2)  # (batch_size, num_points, 2)

        if prev_control_points is not None:
            control_points = prev_control_points + delta_control_points
        else:
            control_points = self.initial_control_points.unsqueeze(0) + delta_control_points  # (batch_size, num_points, 2)

        grid = self.generate_tps_grid(control_points, t.size(2), t.size(3))  # (batch_size, H, W, 2)
        transformed_t = F.grid_sample(t, grid, mode='bilinear', padding_mode='border', align_corners=True)

        return transformed_t, control_points

    def generate_tps_grid(self, control_points, out_h, out_w):
        dtype = control_points.dtype
        device = control_points.device
        control_points = control_points.double()

        batch_size = control_points.size(0)
        num_points = self.num_points

        P = self.initial_control_points.unsqueeze(0).expand(batch_size, num_points, 2).double()  # (batch_size, num_points, 2)

        pairwise_diff = P.unsqueeze(2) - P.unsqueeze(1)  # (batch_size, num_points, num_points, 2)
        pairwise_dist = torch.norm(pairwise_diff, dim=3)  # (batch_size, num_points, num_points)

        epsilon = 1e-6
        K = pairwise_dist ** 2 * torch.log(pairwise_dist ** 2 + epsilon)
        ones = torch.ones(batch_size, num_points, 1, dtype=torch.double, device=device)
        zeros = torch.zeros(batch_size, 3, 3, dtype=torch.double, device=device)
        P_aug = torch.cat([ones, P], dim=2)  # (batch_size, num_points, 3)
        L_top = torch.cat([K, P_aug], dim=2)  # (batch_size, num_points, num_points + 3)
        L_bottom = torch.cat([P_aug.transpose(1, 2), zeros], dim=2)  # (batch_size, 3, num_points + 3)
        L = torch.cat([L_top, L_bottom], dim=1)  # (batch_size, num_points + 3, num_points + 3)

        target_control_points = control_points  # (batch_size, num_points, 2)
        Y = torch.cat([target_control_points, torch.zeros(batch_size, 3, 2, dtype=torch.double, device=device)], dim=1)  # (batch_size, num_points + 3, 2)

        W = torch.linalg.pinv(L).bmm(Y)  # (batch_size, num_points + 3, 2)

        grid_x = torch.linspace(-1, 1, steps=out_w, device=device)
        grid_y = torch.linspace(-1, 1, steps=out_h, device=device)
        grid_p, grid_q = torch.meshgrid(grid_x, grid_y, indexing='ij')  # (out_w, out_h)
        grid = torch.stack([grid_p.flatten(), grid_q.flatten()], dim=1)  # (out_w * out_h, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, out_w * out_h, 2)

        diff = grid.unsqueeze(2) - P.unsqueeze(1)  # (batch_size, out_w * out_h, num_points, 2)
        r = torch.norm(diff, dim=3)  # (batch_size, out_w * out_h, num_points)
        U = r ** 2 * torch.log(r ** 2 + epsilon)

        ones = torch.ones(batch_size, grid.size(1), 1, dtype=torch.double, device=device)
        grid_aug = torch.cat([U, ones, grid], dim=2)  # (batch_size, out_w * out_h, num_points + 3)
        grid_new = grid_aug.bmm(W)  # (batch_size, out_w * out_h, 2)
        grid_new = grid_new.view(batch_size, out_w, out_h, 2).permute(0, 2, 1, 3)  # (batch_size, out_h, out_w, 2)
        grid_new = grid_new.to(dtype)
        return grid_new  # (batch_size, out_h, out_w, 2)

class CascadedSpatialFeatureAlignmentNetwork(nn.Module):
    def __init__(self, in_ch_list, grid_size=5):
        super(CascadedSpatialFeatureAlignmentNetwork, self).__init__()
        self.alignment_modules = nn.ModuleList()
        for in_ch in in_ch_list:
            self.alignment_modules.append(SpatialFeatureAlignmentNetwork(in_ch, grid_size))

    def forward(self, features_t, features_rgb):
        aligned_features = []
        prev_control_points = None
        for i, module in enumerate(self.alignment_modules):
            t_feat = features_t[i]
            rgb_feat = features_rgb[i]
            if prev_control_points is not None:
                aligned_t, current_control_points = module([t_feat, rgb_feat], prev_control_points)
            else:
                aligned_t, current_control_points = module([t_feat, rgb_feat])

            aligned_features.append(aligned_t)
            prev_control_points = current_control_points

        return aligned_features