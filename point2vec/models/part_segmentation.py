from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

from point2vec.modules.feature_upsampling import PointNetFeatureUpsampling
from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder, TransformerEncoderOutput
from point2vec.utils import transforms
from point2vec.utils.checkpoint import extract_model_checkpoint


class Point2VecPartSegmentation(pl.LightningModule):
    def __init__(
        self,
        tokenizer_num_groups: int = 128,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        encoder_unfreeze_epoch: int = 0,
        seg_head_fetch_layers: List[int] = [3, 7, 11],
        seg_head_dim: int = 512,
        seg_head_dropout: float = 0.5,
        learning_rate: float = 0.001,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 10,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        pretrained_ckpt_path: str | None = None,
        train_transformations: List[str] = [
            "center",
            "unit_sphere",
        ],  # scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = ["center", "unit_sphere"],
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [1],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        def build_transformation(name: str) -> transforms.Transform:
            if name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [build_transformation(name) for name in val_transformations]
        )

        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_num_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_group_radius,
            token_dim=encoder_dim,
        )

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )

        dpr = [
            x.item() for x in torch.linspace(0, encoder_drop_path_rate, encoder_depth)
        ]
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            qkv_bias=True,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        self.loss_func = nn.NLLLoss()

    def setup(self, stage: Optional[str] = None) -> None:
        self.num_classes: int = self.trainer.datamodule.num_classes  # type: ignore
        self.num_seg_classes: int = self.trainer.datamodule.num_seg_classes  # type: ignore
        self.category_to_seg_classes: Dict[str, List[int]] = self.trainer.datamodule.category_to_seg_classes  # type: ignore
        self.seg_class_to_category: Dict[int, str] = self.trainer.datamodule.seg_class_to_category  # type: ignore

        label_embedding_dim = 64
        self.label_embedding = nn.Sequential(
            nn.Linear(self.num_classes, label_embedding_dim, bias=False),
            nn.BatchNorm1d(label_embedding_dim),
            nn.LeakyReLU(0.2),
        )

        point_dim = 3
        upsampling_dim = 384
        self.upsampling = PointNetFeatureUpsampling(in_channel=self.hparams.encoder_dim + point_dim, mlp=[upsampling_dim, upsampling_dim])  # type: ignore

        self.seg_head = nn.Sequential(
            nn.Conv1d(
                2 * self.hparams.encoder_dim + label_embedding_dim + upsampling_dim,  # type: ignore
                self.hparams.seg_head_dim,  # type: ignore
                1,
                bias=False,
            ),
            nn.BatchNorm1d(self.hparams.seg_head_dim),  # type: ignore
            nn.ReLU(),
            nn.Dropout(self.hparams.seg_head_dropout),  # type: ignore
            nn.Conv1d(self.hparams.seg_head_dim, self.hparams.seg_head_dim // 2, 1, bias=False),  # type: ignore
            nn.BatchNorm1d(self.hparams.seg_head_dim // 2),  # type: ignore
            nn.ReLU(),
            # nn.Dropout(self.hparams.seg_head_dropout),
            nn.Conv1d(self.hparams.seg_head_dim // 2, self.num_seg_classes, 1),  # type: ignore
        )

        self.val_macc = Accuracy(num_classes=self.num_seg_classes, average="macro")

        if self.hparams.pretrained_ckpt_path is not None:  # type: ignore
            self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)  # type: ignore

        self.encoder.requires_grad_(False)  # will unfreeze later

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")
                logger.experiment.define_metric("val_ins_miou", summary="last,max")
                logger.experiment.define_metric("val_cat_miou", summary="last,max")

    def forward(self, points: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        # label:  (B,)
        B, N, C = points.shape

        tokens: torch.Tensor
        centers: torch.Tensor
        tokens, centers = self.tokenizer(points)  # (B, T, C), (B, T, 3)
        pos_embeddings = self.positional_encoding(centers)
        output: TransformerEncoderOutput = self.encoder(
            tokens, pos_embeddings, return_hidden_states=True
        )
        hidden_states = [F.layer_norm((output.hidden_states[i]), output.hidden_states[i].shape[-1:]) for i in self.hparams.seg_head_fetch_layers]  # type: ignore [(B, T, C)]
        token_features = torch.stack(hidden_states, dim=0).mean(0)  # (B, T, C)
        token_features_max = token_features.max(dim=1).values  # (B, C)
        token_features_mean = token_features.mean(dim=1)  # (B, C)

        label_embedding = self.label_embedding(self.categorical_label(label))  # (B, L)

        global_feature = torch.cat(
            [token_features_max, token_features_mean, label_embedding], dim=-1
        )  # (B, 2*C' + L)

        x = self.upsampling(points, centers, points, token_features)  # (B, N, C)
        x = torch.cat(
            [x, global_feature.unsqueeze(-1).expand(-1, -1, N).transpose(1, 2)], dim=-1
        )  # (B, N, C'); C' = 3*C + L
        x = self.seg_head(x.transpose(1, 2)).transpose(1, 2)  # (B, N, cls)
        x = F.log_softmax(x, dim=-1)
        return x

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N, 3)
        # seg_labels: (B, N)
        # label: (B,)
        points, seg_labels, label = batch
        points = self.train_transformations(points)
        log_probablities = self.forward(points, label)

        log_probablities = log_probablities.view(-1, self.num_seg_classes)
        seg_labels = seg_labels.view(-1)
        loss = self.loss_func(log_probablities, seg_labels)
        self.log("train_loss", loss, on_epoch=True)

        pred = torch.max(log_probablities, dim=-1).indices
        self.train_acc(pred, seg_labels)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, List[torch.Tensor]]:
        # points: (B, N, 3)
        # seg_labels: (B, N)
        # label: (B,)
        points, seg_labels, label = batch
        points = self.val_transformations(points)
        log_probablities = self.forward(points, label)

        ious = self.compute_shape_ious(log_probablities, seg_labels)

        log_probablities = log_probablities.view(-1, self.num_seg_classes)
        seg_labels = seg_labels.view(-1)
        loss = self.loss_func(log_probablities, seg_labels)
        self.log("val_loss", loss)

        pred = torch.max(log_probablities, dim=-1).indices
        self.val_acc(pred, seg_labels)
        self.log("val_acc", self.val_acc)
        self.val_macc(pred, seg_labels)
        self.log("val_macc", self.val_macc)

        return ious

    def validation_epoch_end(
        self, outputs: List[Dict[str, List[torch.Tensor]]]
    ) -> None:
        shape_mious: Dict[str, List[torch.Tensor]] = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }
        for d in outputs:
            for k, v in d.items():
                shape_mious[k] = shape_mious[k] + v

        all_shape_mious = torch.stack(
            [miou for mious in shape_mious.values() for miou in mious]
        )
        cat_mious = {
            k: torch.stack(v).mean() for k, v in shape_mious.items() if len(v) > 0
        }

        self.log("val_ins_miou", all_shape_mious.mean())
        self.log("val_cat_miou", torch.stack(list(cat_mious.values())).mean())
        for cat in sorted(cat_mious.keys()):
            self.log(f"val_cat_miou_{cat}", cat_mious[cat])

    def compute_shape_ious(
        self, log_probablities: torch.Tensor, seg_labels: torch.Tensor
    ) -> Dict[str, List[torch.Tensor]]:
        # log_probablities: (B, N, 50) \in -inf..<0
        # seg_labels:       (B, N) \in 0..<50
        # returns           { cat: (S, P) }

        shape_ious: Dict[str, List[torch.Tensor]] = {
            cat: [] for cat in self.category_to_seg_classes.keys()
        }

        for i in range(log_probablities.shape[0]):
            cat = self.seg_class_to_category[seg_labels[i, 0].item()]  # type: ignore
            seg_classes = self.category_to_seg_classes[cat]
            seg_preds = (
                torch.argmax(
                    log_probablities[i, :, self.category_to_seg_classes[cat]], dim=1
                )
                + seg_classes[0]
            )  # (N,)

            seg_class_iou = torch.empty(len(seg_classes))
            for c in seg_classes:
                if ((seg_labels[i] == c).sum() == 0) and (
                    (seg_preds == c).sum() == 0
                ):  # part is not present, no prediction as well
                    seg_class_iou[c - seg_classes[0]] = 1.0
                else:
                    intersection = ((seg_labels[i] == c) & (seg_preds == c)).sum()
                    union = ((seg_labels[i] == c) | (seg_preds == c)).sum()
                    seg_class_iou[c - seg_classes[0]] = intersection / union
            shape_ious[cat].append(seg_class_iou.mean())

        return shape_ious

    def configure_optimizers(self):
        assert self.trainer is not None

        opt = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=self.hparams.lr_scheduler_linear_warmup_epochs,  # type: ignore
            max_epochs=self.trainer.max_epochs,
            warmup_start_lr=self.hparams.lr_scheduler_linear_warmup_start_lr,  # type: ignore
            eta_min=self.hparams.lr_scheduler_cosine_eta_min,  # type: ignore
        )
        return [opt], [sched]

    def load_pretrained_checkpoint(self, path: str) -> None:
        print(f"Loading pretrained checkpoint from '{path}'.")

        checkpoint = extract_model_checkpoint(path)

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

    def on_train_epoch_start(self) -> None:
        if self.trainer.current_epoch == self.hparams.encoder_unfreeze_epoch:  # type: ignore
            self.encoder.requires_grad_(True)
            print("Unfreeze encoder")

    def categorical_label(self, label: torch.Tensor) -> torch.Tensor:
        # label: (B,)
        return torch.eye(self.num_classes, device=label.device)[label]
