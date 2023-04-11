from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

from point2vec.modules.pointnet import PointcloudTokenizer
from point2vec.modules.transformer import TransformerEncoder
from point2vec.utils import transforms
from point2vec.utils.checkpoint import extract_model_checkpoint


class Point2VecClassification(pl.LightningModule):
    def __init__(
        self,
        num_points: int = 1024,
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_unfreeze_epoch: int = 0,
        positional_encoding_unfreeze_epoch: int = 0,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        encoder_qkv_bias: bool = True,
        encoder_freeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_epoch: int = 0,
        encoder_unfreeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_stepwise: bool = False,
        encoder_unfreeze_stepwise_num_layers: int = 2,
        encoder_learning_rate: Optional[float] = None,
        cls_head: str = "mlp",  # mlp, linear
        cls_head_dim: int = 256,
        cls_head_dropout: float = 0.5,
        cls_head_pooling: str = "mean+max",  # mean+max+cls_token, mean+max, mean, max, cls_token
        loss_label_smoothing: float = 0.2,
        learning_rate: float = 0.001,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 10,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        pretrained_ckpt_path: str | None = None,
        pretrained_ckpt_ignore_encoder_layers: List[int] = [],
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
        log_tsne: bool = False,
        log_confusion_matrix: bool = False,
        vote: bool = False,
        vote_num: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        if vote:
            self.val_vote_acc = Accuracy()

        self.val_top_3_acc = Accuracy(top_k=3)

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
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in val_transformations]
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
            qkv_bias=encoder_qkv_bias,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attention_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=encoder_add_pos_at_every_layer,
        )

        match cls_head_pooling:
            case "cls_token" | "mean+max+cls_token":
                init_std = 0.02
                self.cls_token = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_token, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.cls_pos = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_pos, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.first_cls_head_dim = (
                    encoder_dim if cls_head_pooling == "cls_token" else 3 * encoder_dim
                )
            case "mean+max":
                self.first_cls_head_dim = 2 * encoder_dim
            case "mean":
                self.first_cls_head_dim = encoder_dim
            case "max":
                self.first_cls_head_dim = encoder_dim
            case _:
                raise ValueError()

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=loss_label_smoothing)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.cls_head == "mlp":  # type: ignore
            self.cls_head = nn.Sequential(
                nn.Linear(
                    self.first_cls_head_dim, self.hparams.cls_head_dim, bias=False  # type: ignore
                ),  # bias can be False because of batch norm following
                nn.BatchNorm1d(self.hparams.cls_head_dim),  # type: ignore
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),  # type: ignore
                nn.Linear(self.hparams.cls_head_dim, self.hparams.cls_head_dim, bias=False),  # type: ignore
                nn.BatchNorm1d(self.hparams.cls_head_dim),  # type: ignore
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),  # type: ignore
                nn.Linear(self.hparams.cls_head_dim, self.trainer.datamodule.num_classes),  # type: ignore
            )
        elif self.hparams.cls_head == "linear":  # type: ignore
            self.cls_head = nn.Linear(self.first_cls_head_dim, self.trainer.datamodule.num_classes)  # type: ignore

        self.val_macc = Accuracy(num_classes=self.trainer.datamodule.num_classes, average="macro")  # type: ignore

        if self.hparams.pretrained_ckpt_path is not None:  # type: ignore
            self.load_pretrained_checkpoint(self.hparams.pretrained_ckpt_path)  # type: ignore

        # freeze encoder here, will unfreeze again in `on_train_epoch_start`
        if self.hparams.encoder_freeze_layers is not None:  # type:ignore
            assert self.hparams.encoder_unfreeze_stepwise == False  # type: ignore
            assert isinstance(self.encoder, TransformerEncoder)
            for i in self.hparams.encoder_freeze_layers:  # type:ignore
                self.encoder.blocks[i].requires_grad_(False)
        else:
            self.encoder.requires_grad_(False)

        self.tokenizer.requires_grad_(False)
        self.positional_encoding.requires_grad_(False)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_top_3_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        tokens: torch.Tensor  # (B, T, C)
        centers: torch.Tensor  # (B, T, 3)
        tokens, centers = self.tokenizer(points)
        pos_embeddings = self.positional_encoding(centers)
        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:  # type: ignore
            B, T, C = tokens.shape
            tokens = torch.cat(
                [self.cls_token.reshape(1, 1, C).expand(B, -1, -1), tokens], dim=1
            )
            pos_embeddings = torch.cat(
                [self.cls_pos.reshape(1, 1, C).expand(B, -1, -1), pos_embeddings], dim=1
            )
        tokens = self.encoder(tokens, pos_embeddings).last_hidden_state
        match self.hparams.cls_head_pooling:  # type: ignore
            case "cls_token":
                embedding = tokens[:, 0]
            case "mean+max":
                max_features = torch.max(tokens, dim=1).values
                mean_features = torch.mean(tokens, dim=1)
                embedding = torch.cat([max_features, mean_features], dim=-1)
            case "mean+max+cls_token":
                cls_token = tokens[:, 0]
                max_features = torch.max(tokens[:, 1:], dim=1).values
                mean_features = torch.mean(tokens[:, 1:], dim=1)
                embedding = torch.cat([cls_token, max_features, mean_features], dim=-1)
            case "mean":
                embedding = torch.mean(tokens, dim=1)
            case "max":
                embedding = torch.max(tokens, dim=1).values
            case _:
                raise ValueError(f"Unknown cls_head_pooling: {self.hparams.cls_head_pooling}")  # type: ignore
        logits = (
            self.cls_head(embedding)  # type: ignore
            if isinstance(tokens, torch.Tensor)
            else self.cls_head(embedding.F)  # type: ignore
        )
        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N, 3)
        # label: (B,)
        points, label = batch
        points = self.train_transformations(points)
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("train_loss", loss, on_epoch=True)

        pred = torch.max(logits, dim=-1).indices
        self.train_acc(pred, label)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # points: (B, N, 3)
        # label: (B,)
        points, label = batch
        points = self.val_transformations(points)
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("val_loss", loss)

        pred = torch.max(logits, dim=-1).indices
        self.val_acc(pred, label)
        self.log("val_acc", self.val_acc)
        self.val_top_3_acc(logits, label)
        self.log("val_top_3_acc", self.val_top_3_acc)
        self.val_macc(pred, label)
        self.log("val_macc", self.val_macc)

        if self.hparams.vote:  # type: ignore
            logits_list = []
            for _ in range(self.hparams.vote_num):  # type: ignore
                points = self.val_transformations(points)
                logits = self.forward(points)
                logits_list.append(F.softmax(logits, dim=-1))
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)
            vote_pred = torch.max(mean_logits, dim=-1).indices

            self.val_vote_acc(vote_pred, label)
            self.log("val_vote_acc", self.val_vote_acc)

    def configure_optimizers(self):
        assert self.trainer is not None

        encoder_params = []
        other_params = []
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        enc_lr: Optional[float] = self.hparams.encoder_learning_rate  # type: ignore
        opt = torch.optim.AdamW(
            params=[
                {"params": encoder_params, "lr": enc_lr if enc_lr is not None else self.hparams.learning_rate},  # type: ignore
                {"params": other_params},
            ],
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

        for k in list(checkpoint.keys()):
            if k.startswith("cls_head."):
                del checkpoint[k]
            elif k.startswith("encoder.blocks."):
                for i in self.hparams.pretrained_ckpt_ignore_encoder_layers:  # type: ignore
                    if k.startswith(f"encoder.blocks.{i}."):
                        del checkpoint[k]

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)  # type: ignore
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # fix NaNs in batchnorm, this has been observed in some checkpoints... not sure why
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm1d):
                if torch.any(torch.isnan(m.running_mean)):  # type: ignore
                    print(f"Warning: NaNs in running_mean of {name}. Setting to zeros.")
                    m.running_mean = torch.zeros_like(m.running_mean)  # type: ignore
                if torch.any(torch.isnan(m.running_var)):  # type: ignore
                    print(f"Warning: NaNs in running_var of {name}. Setting to ones.")
                    m.running_var = torch.ones_like(m.running_var)  # type: ignore

    def on_train_epoch_start(self) -> None:
        if self.hparams.encoder_unfreeze_stepwise:  # type: ignore
            assert isinstance(self.encoder, TransformerEncoder)
            assert self.hparams.encoder_unfreeze_epoch >= 0  # type: ignore
            unfreeze_epochs = list(
                range(
                    self.hparams.encoder_unfreeze_epoch,  # type: ignore
                    self.trainer.max_epochs,
                    int(
                        (
                            self.trainer.max_epochs
                            - self.hparams.encoder_unfreeze_epoch  # type:ignore
                        )
                        / (
                            (
                                self.hparams.encoder_depth  # type: ignore
                                if self.hparams.encoder_unfreeze_layers is None  # type: ignore
                                else len(
                                    self.hparams.encoder_unfreeze_layers  # type:ignore
                                )
                            )
                            / self.hparams.encoder_unfreeze_stepwise_num_layers  # type: ignore
                        )
                    ),
                )
            )
            if self.trainer.current_epoch in unfreeze_epochs:
                i = unfreeze_epochs.index(self.trainer.current_epoch)
                for j in range(
                    i * self.hparams.encoder_unfreeze_stepwise_num_layers, (i + 1) * self.hparams.encoder_unfreeze_stepwise_num_layers  # type: ignore
                ):
                    unfreeze_layer_idx = -(j + 1)
                    if self.hparams.encoder_unfreeze_layers is not None:  # type: ignore
                        unfreeze_layer_idx: int = self.hparams.encoder_unfreeze_layers[unfreeze_layer_idx]  # type: ignore
                    self.encoder.blocks[unfreeze_layer_idx].requires_grad_(True)
                    self.print(f"Unfreeze encoder layer {unfreeze_layer_idx}")
        elif self.trainer.current_epoch == self.hparams.encoder_unfreeze_epoch:  # type: ignore
            if self.hparams.encoder_unfreeze_layers is not None:  # type:ignore
                assert isinstance(self.encoder, TransformerEncoder)
                for i in self.hparams.encoder_unfreeze_layers:  # type:ignore
                    self.encoder.blocks[i].requires_grad_(True)
                    print(f"Unfreeze encoder layer {i}")
            else:
                self.encoder.requires_grad_(True)
                print("Unfreeze encoder")

        if self.trainer.current_epoch == self.hparams.tokenizer_unfreeze_epoch:  # type: ignore
            self.tokenizer.requires_grad_(True)
            print("Unfreeze tokenizer")
        if self.trainer.current_epoch == self.hparams.positional_encoding_unfreeze_epoch:  # type: ignore
            self.positional_encoding.requires_grad_(True)
            print("Unfreeze positional encoding")
