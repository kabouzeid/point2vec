from typing import Callable, Optional, Sequence, Tuple

import torch
from pytorch3d.ops import sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from pytorch3d.transforms import euler_angles_to_matrix
from torch import nn

Transform = Callable[[torch.Tensor], torch.Tensor]


def resample_points(points: torch.Tensor, num_points: int) -> torch.Tensor:
    if points.shape[1] > num_points:
        if num_points == 1024:
            num_samples = 1200
        elif num_points == 2048:
            num_samples = 2400
        elif num_points == 4096:
            num_samples = 4800
        elif num_points == 8192:
            num_samples = 8192
        else:
            raise NotImplementedError()
        if points.shape[1] < num_samples:
            num_samples = points.shape[1]
        _, idx = sample_farthest_points(
            points[:, :, :3].float(), K=num_samples, random_start_point=True
        )
        points = masked_gather(points, idx)
        points = points[:, torch.randperm(num_samples)[:num_points]]
        return points
    else:
        raise RuntimeError("Not enough points")


class PointcloudSubsampling(nn.Module):
    def __init__(self, num_points: int, strategy: str = "fps"):
        super().__init__()
        self.num_points = num_points
        self.strategy = strategy

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        if points.shape[1] < self.num_points:
            raise RuntimeError(
                f"Too few points in pointcloud: {points.shape[1]} vs {self.num_points}"
            )
        elif points.shape[1] == self.num_points:
            return points

        if self.strategy == "resample":
            return resample_points(points, self.num_points)
        elif self.strategy == "fps":
            _, idx = sample_farthest_points(
                points[:, :, :3].float(), K=self.num_points, random_start_point=True
            )
            return masked_gather(points, idx)
        elif self.strategy == "random":
            return points[:, torch.randperm(points.shape[1])[: self.num_points]]
        else:
            raise RuntimeError(f"No such subsampling strategy {self.strategy}")


# TODO: remove this
class PointcloudCenterAndNormalize(nn.Module):
    def __init__(
        self,
        centering: bool = True,
        normalize=True,
    ):
        super().__init__()
        self.centering = centering
        self.normalize = normalize

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        if self.centering:
            points[:, :, :3] = points[:, :, :3] - torch.mean(
                points[:, :, :3], dim=-2, keepdim=True
            )
        if self.normalize:
            max_norm = torch.max(
                torch.norm(points[:, :, :3], dim=-1, keepdim=True),
                dim=-2,
                keepdim=True,
            ).values
            points[:, :, :3] = points[:, :, :3] / max_norm
        return points


class PointcloudCentering(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        points[:, :, :3] = points[:, :, :3] - torch.mean(
            points[:, :, :3], dim=-2, keepdim=True
        )
        return points


class PointcloudUnitSphere(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        max_norm = torch.max(
            torch.norm(points[:, :, :3], dim=-1, keepdim=True),
            dim=-2,
            keepdim=True,
        ).values
        points[:, :, :3] = points[:, :, :3] / max_norm
        return points


class PointcloudHeightNormalization(nn.Module):
    def __init__(
        self,
        dim: int,
        append: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.append = append

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        min_height = torch.min(points[:, :, self.dim], dim=-1).values
        heights = points[:, :, self.dim] - min_height.unsqueeze(-1)
        if self.append:
            points = torch.cat([points, heights.unsqueeze(-1)], dim=-1)
        else:
            points[:, :, self.dim] = heights
        return points


class PointcloudScaling(nn.Module):
    def __init__(
        self,
        min: float,
        max: float,
        anisotropic: bool = True,
        scale_xyz: Tuple[bool, bool, bool] = (True, True, True),
        symmetries: Tuple[int, int, int] = (0, 0, 0),  # mirror scaling, x --> -x
    ):
        super().__init__()
        self.scale_min = min
        self.scale_max = max
        self.anisotropic = anisotropic
        self.scale_xyz = scale_xyz
        self.symmetries = torch.tensor(symmetries)

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        scale = (
            torch.rand(3 if self.anisotropic else 1, device=points.device)
            * (self.scale_max - self.scale_min)
            + self.scale_min
        )

        symmetries = torch.round(torch.rand(3, device=points.device)) * 2 - 1
        self.symmetries = self.symmetries.to(points.device)
        symmetries = symmetries * self.symmetries + (1 - self.symmetries)
        scale *= symmetries
        for i, s in enumerate(self.scale_xyz):
            if not s:
                scale[i] = 1
        points[:, :, :3] = points[:, :, :3] * scale
        return points


class PointcloudTranslation(nn.Module):
    def __init__(
        self,
        translation: float,
    ):
        super().__init__()
        self.translation = translation

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        translation = (
            torch.rand(3, device=points.device) * 2 * self.translation
            - self.translation
        )

        points[:, :, :3] = points[:, :, :3] + translation
        return points


class PointcloudRotation(nn.Module):
    def __init__(self, dims: Sequence[int], deg: Optional[int] = None):
        # deg: \in [0...179], eg 45 means rotation steps of 45 deg are allowed
        super().__init__()
        self.dims = dims
        self.deg = deg
        assert self.deg is None or (self.deg >= 0 and self.deg <= 180)

    def forward(self, points: torch.Tensor):
        # points: (B, N, 3)
        euler_angles = torch.zeros(3)
        for dim in self.dims:
            if self.deg is not None:
                possible_degs = (
                    torch.tensor(list(range(0, 360, self.deg))) / 360
                ) * 2 * torch.pi - torch.pi
                euler_angles[dim] = possible_degs[
                    torch.randint(high=len(possible_degs), size=(1,))
                ]
            else:
                euler_angles[dim] = (2 * torch.pi) * torch.rand(1) - torch.pi
        R = euler_angles_to_matrix(euler_angles, "XYZ").to(points.device)
        points[:, :, :3] = points[:, :, :3] @ R.T
        return points


class Compose:
    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = transforms

    def __call__(self, points: torch.Tensor):
        for t in self.transforms:
            points = t(points)
        return points
