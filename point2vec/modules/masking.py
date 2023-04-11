import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from torch import nn


class PointcloudMasking(nn.Module):
    def __init__(self, ratio: float, type: str):
        super().__init__()
        self.ratio = ratio

        if type == "rand":
            self.forward = self._mask_center_rand
        elif type == "block":
            self.forward = self._mask_center_block
        else:
            raise ValueError(f"No such masking type: {type}")

    def _mask_center_rand(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, _ = centers.shape

        num_mask = int(self.ratio * G)

        mask = (
            torch.cat(
                [
                    torch.zeros(G - num_mask, device=centers.device),
                    torch.ones(num_mask, device=centers.device),
                ]
            )
            .to(torch.bool)
            .unsqueeze(0)
            .expand(B, -1)
        )  # (B, G)
        # TODO: profile if this loop is slow
        for i in range(B):
            mask[i, :] = mask[i, torch.randperm(mask.shape[1])]

        return mask  # (B, G)

    def _mask_center_block(self, centers: torch.Tensor) -> torch.Tensor:
        # centers: (B, G, 3)
        if self.ratio == 0:
            return torch.zeros(centers.shape[:2]).bool()

        B, G, D = centers.shape
        assert D == 3

        num_mask = int(self.ratio * G)

        # random center
        center = torch.empty((B, 1, D), device=centers.device)
        for i in range(B):
            center[i, 0, :] = centers[i, torch.randint(0, G, (1,)), :]

        # center's nearest neighbors
        _, knn_idx, _ = knn_points(
            center.float(), centers.float(), K=num_mask, return_sorted=False
        )  # (B, 1, K)
        knn_idx = knn_idx.squeeze(1)  # (B, K)

        mask = torch.zeros([B, G], device=centers.device)
        mask.scatter_(dim=1, index=knn_idx, value=1.0)
        mask = mask.to(torch.bool)
        return mask
