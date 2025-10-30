import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedMLP(nn.Module):
    """
    PointNet-style shared MLP implemented with 1x1 Conv1d + BatchNorm + ReLU.
    Expects input of shape [B, C_in, N] and outputs [B, C_out, N].
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class SimplePointNetCPL(nn.Module):
    """
    SimplePointNet-CPL

    Hierarchical PointNet-style model with a deterministic Critical Points Layer (CPL)
    for down-sampling from 1024 to 256 critical points.

    Stages:
      1) Initial Feature Embedding: shared MLPs (64, 128, 1024)
      2) CPL down-sampling to 256 points using max-contribution-based selection
      3) Final Feature Extraction: shared MLP (1024)
      4) Classification head: 1024 -> 512 -> 256 -> 40

    Input: [B, 1024, 3]
    Output: logits [B, 40]
    """

    def __init__(self, num_points: int = 1024, num_classes: int = 40, cpl_points: int = 256,
                 dropout_p: float = 0.4):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.cpl_points = cpl_points

        # Part 1: Initial Feature Embedding (PointNet shared MLP backbone)
        self.mlp1 = SharedMLP(3, 64)
        self.mlp2 = SharedMLP(64, 128)
        self.mlp3 = SharedMLP(128, 1024)

        # Part 3: Final Feature Extraction over critical points
        self.mlp_critical = SharedMLP(1024, 1024)

        # Classification Head
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout_p)

        self.fc3 = nn.Linear(256, num_classes)

    @staticmethod
    def _resize_indices_deterministic(indices: torch.Tensor, target_k: int) -> torch.Tensor:
        m = indices.shape[0]
        if m == 0:
            return torch.zeros(target_k, dtype=indices.dtype, device=indices.device)
        if m == target_k:
            return indices
        if m > target_k:
            return indices[:target_k]
        pos = torch.linspace(0, max(m - 1, 0), steps=target_k, device=indices.device)
        nn_pos = torch.round(pos).long()
        nn_pos = torch.clamp(nn_pos, 0, m - 1)
        return indices[nn_pos]

    def _cpl_downsample(self, features_bnc: torch.Tensor) -> torch.Tensor:
        bsz, num_points, _ = features_bnc.shape
        features_bcn = features_bnc.transpose(1, 2)
        max_vals, max_idx = torch.max(features_bcn, dim=2)
        selected_feats = []
        for b in range(bsz):
            idx_b = max_idx[b]
            vals_b = max_vals[b]
            scores = torch.zeros(num_points, device=features_bnc.device, dtype=features_bnc.dtype)
            scores = scores.scatter_add(0, idx_b, vals_b)
            eps = 1e-7
            adjusted = -scores + eps * (torch.arange(num_points, device=scores.device, dtype=scores.dtype))
            sorted_all = torch.argsort(adjusted, dim=0, stable=True)
            contributing_mask = scores > 0
            if torch.any(contributing_mask):
                contributing_sorted = sorted_all[contributing_mask[sorted_all]]
            else:
                contributing_sorted = sorted_all
            resized_idx = self._resize_indices_deterministic(contributing_sorted, self.cpl_points)
            feats_b = features_bnc[b]
            gathered = feats_b.index_select(dim=0, index=resized_idx)
            selected_feats.append(gathered)
        return torch.stack(selected_feats, dim=0)

    def _cpl_indices(self, features_bnc: torch.Tensor) -> torch.Tensor:
        bsz, num_points, _ = features_bnc.shape
        features_bcn = features_bnc.transpose(1, 2)
        max_vals, max_idx = torch.max(features_bcn, dim=2)
        batch_indices = []
        for b in range(bsz):
            idx_b = max_idx[b]
            vals_b = max_vals[b]
            scores = torch.zeros(num_points, device=features_bnc.device, dtype=features_bnc.dtype)
            scores = scores.scatter_add(0, idx_b, vals_b)
            eps = 1e-7
            adjusted = -scores + eps * (torch.arange(num_points, device=scores.device, dtype=scores.dtype))
            sorted_all = torch.argsort(adjusted, dim=0, stable=True)
            contributing_mask = scores > 0
            if torch.any(contributing_mask):
                contributing_sorted = sorted_all[contributing_mask[sorted_all]]
            else:
                contributing_sorted = sorted_all
            resized_idx = self._resize_indices_deterministic(contributing_sorted, self.cpl_points)
            batch_indices.append(resized_idx)
        return torch.stack(batch_indices, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[2] == 3
        bsz, n_pts, _ = x.shape
        assert n_pts == self.num_points
        x_t = x.transpose(1, 2)
        x_feat = self.mlp1(x_t)
        x_feat = self.mlp2(x_feat)
        x_feat = self.mlp3(x_feat)
        features_bnc = x_feat.transpose(1, 2)
        critical_bkc = self._cpl_downsample(features_bnc)
        crit_cfb = critical_bkc.transpose(1, 2)
        crit_cfb = self.mlp_critical(crit_cfb)
        global_feat = torch.max(crit_cfb, dim=2)[0]
        x = self.fc1(global_feat)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dp2(x)
        logits = self.fc3(x)
        return logits

    @torch.no_grad()
    def extract_cpl(self, x: torch.Tensor):
        assert x.dim() == 3 and x.shape[2] == 3
        x_t = x.transpose(1, 2)
        x_feat = self.mlp1(x_t)
        x_feat = self.mlp2(x_feat)
        x_feat = self.mlp3(x_feat)
        features_bnc = x_feat.transpose(1, 2)
        idx = self._cpl_indices(features_bnc)
        bsz, _, _ = features_bnc.shape
        k = idx.shape[1]
        batch_arange = torch.arange(bsz, device=features_bnc.device).view(bsz, 1).expand(bsz, k)
        critical_features = features_bnc[batch_arange, idx]
        return critical_features, idx


def build_simple_pointnet_cpl(num_points: int = 1024, num_classes: int = 40, cpl_points: int = 256,
                              dropout_p: float = 0.4) -> SimplePointNetCPL:
    return SimplePointNetCPL(num_points=num_points, num_classes=num_classes,
                             cpl_points=cpl_points, dropout_p=dropout_p)


