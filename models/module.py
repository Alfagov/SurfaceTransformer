from pathlib import Path

import torch
import lightning as pl
from torch.optim import Adam, AdamW
import torch.nn as nn
from pytorch_optimizer import SOAP

from models.loss import GreeksInformedLoss
from models.net import SurfaceTransformerNet
from models.utils import _build_surface_example_input

from torch.optim.lr_scheduler import CosineAnnealingLR

class SurfaceTransformerOptionModule(pl.LightningModule):
    def __init__(
            self,
            raw_input_dim: int = 6,
            context_size: int = 64,
            query_size: int = 64,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 4,
            ff_dim: int = 256,
            head_hidden: int = 256,
            lambda_arb: float = 1.0,
            theta_floor_base: float = -0.03,
            theta_floor_slope: float = 0.0393,
            theta_floor_eps: float = 1e-4,
            delta_ceiling: float = 0.9999,
            price_spot_margin: float = 1e-4,
            w_delta_upper: float = 1.0,
            w_theta_upper: float = 1.0,
            w_price_upper: float = 1.0,
            w_delta_target: float = 1.0,
            w_gamma_target: float = 1.0,
            w_theta_target: float = 1.0,
            vega_weight_eps: float = 1e-8,
            huber_delta: float = 1.0,
            learning_rate: float = 1e-3,
            price_log_eps: float = 1e-6,
            log_price_clip_max: float = 20.0,
            epochs: int = 1000
    ):
        super().__init__()
        self.save_hyperparameters()
        self.epochs = epochs


        self.model = SurfaceTransformerNet(
            raw_input_dim=raw_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            head_hidden=head_hidden,
            price_log_eps=price_log_eps,
            log_price_clip_max=log_price_clip_max,
        )

        self.loss_fn = GreeksInformedLoss(
            lambda_arb=lambda_arb,
            theta_floor_base=theta_floor_base,
            theta_floor_slope=theta_floor_slope,
            theta_floor_eps=theta_floor_eps,
            delta_ceiling=delta_ceiling,
            price_spot_margin=price_spot_margin,
            w_delta_upper=w_delta_upper,
            w_theta_upper=w_theta_upper,
            w_price_upper=w_price_upper,
            w_delta_target=w_delta_target,
            w_gamma_target=w_gamma_target,
            w_theta_target=w_theta_target,
            vega_weight_eps=vega_weight_eps,
            huber_delta=huber_delta,
        )

        self.lambda_arb = lambda_arb
        self.context_size = context_size
        self.query_size = query_size
        self.example_input_array = (
            _build_surface_example_input(
                raw_input_dim=raw_input_dim,
                context_size=context_size,
                query_size=query_size,
            )
        )

    def forward(
            self,
            query_x: torch.Tensor,
            context_x: torch.Tensor,
            context_y: torch.Tensor,
            context_mask: torch.Tensor,
    ):
        return self.model(
            query_x=query_x,
            context_x=context_x,
            context_y=context_y,
            context_mask=context_mask,
        )

    @staticmethod
    def _guard_finite(tensor: torch.Tensor) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        return torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

    def _shared_step(self, batch, stage: str):
        query_x = batch["query_x"]
        query_y = batch["query_y"]
        query_greeks = batch["query_greeks"]
        query_mask = batch.get("query_mask")
        context_x = batch["context_x"]
        context_y = batch["context_y"]
        context_mask = batch["context_mask"]

        y_pred = self(
            query_x=query_x,
            context_x=context_x,
            context_y=context_y,
            context_mask=context_mask,
        )
        t = query_x[:, :, 2:3]
        s = query_x[:, :, 0:1]
        k = query_x[:, :, 1:2]
        loss_dict = self.loss_fn(
            y_pred,
            query_y,
            t=t,
            s=s,
            k=k,
            target_greeks=query_greeks,
            sample_mask=query_mask,
        )
        loss_dict["loss"] = self._guard_finite(loss_dict["loss"])

        self.log(f"{stage}_loss", loss_dict["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{stage}_price_loss", loss_dict["price_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(
            f"{stage}_greeks_loss",
            self.lambda_arb * loss_dict["greek_penalty"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(f"{stage}_delta_loss", loss_dict["delta_loss"], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            f"{stage}_delta_upper_loss",
            loss_dict["delta_upper_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(f"{stage}_gamma_loss", loss_dict["gamma_loss"], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(f"{stage}_theta_loss", loss_dict["theta_loss"], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log(
            f"{stage}_theta_upper_loss",
            loss_dict["theta_upper_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_dual_delta_loss",
            loss_dict["dual_delta_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_dual_gamma_loss",
            loss_dict["dual_gamma_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_price_upper_loss",
            loss_dict["price_upper_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_theta_floor_mean",
            loss_dict["theta_floor_mean"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_delta_target_loss",
            loss_dict["delta_target_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_gamma_target_loss",
            loss_dict["gamma_target_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{stage}_theta_target_loss",
            loss_dict["theta_target_loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        return loss_dict

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def on_validation_model_eval(self) -> None:
        super().on_validation_model_eval()
        torch.set_grad_enabled(True)

    def on_test_model_eval(self) -> None:
        super().on_test_model_eval()
        torch.set_grad_enabled(True)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        optimizer = SOAP(self.parameters(), lr=self.hparams.learning_rate)#Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)#StepLR(optimizer, step_size=10, gamma=0.95)
        return [optimizer], [scheduler]

def export_surface_model_to_onnx(
        surface_module: SurfaceTransformerOptionModule,
        checkpoint_dir: Path,
        run_id: str,
) -> Path:
    ONNX_OPSET_VERSION = 18

    onnx_path = checkpoint_dir / f"surface_transformer_{run_id}.onnx"

    export_model = SurfacePriceExportWrapper(surface_module.cpu()).eval()
    query_x, context_x, context_y, context_mask = surface_module.example_input_array
    query_x = query_x.detach().cpu()
    context_x = context_x.detach().cpu()
    context_y = context_y.detach().cpu()
    context_mask = context_mask.detach().cpu()

    with torch.no_grad():
        batch_dim = torch.export.Dim("batch")
        query_points_dim = torch.export.Dim("query_points")
        context_points_dim = torch.export.Dim("context_points")
        torch.onnx.export(
            export_model,
            (query_x, context_x, context_y, context_mask),
            str(onnx_path),
            export_params=True,
            do_constant_folding=True,
            dynamo=True,
            opset_version=ONNX_OPSET_VERSION,
            input_names=["query_x", "context_x", "context_y", "context_mask"],
            output_names=["predicted_price"],
            dynamic_shapes=(
                {0: batch_dim, 1: query_points_dim},
                {0: batch_dim, 1: context_points_dim},
                {0: batch_dim, 1: context_points_dim},
                {0: batch_dim, 1: context_points_dim},
            ),
        )

    return onnx_path

class SurfacePriceExportWrapper(nn.Module):
    """ONNX-friendly inference wrapper that exports surface call prices only."""

    def __init__(self, surface_module: SurfaceTransformerOptionModule):
        super().__init__()
        self.surface_model = surface_module.model

    def forward(
            self,
            query_x: torch.Tensor,
            context_x: torch.Tensor,
            context_y: torch.Tensor,
            context_mask: torch.Tensor,
    ) -> torch.Tensor:
        context_memory = self.surface_model.encoder(
            context_x=context_x,
            context_y=context_y,
            context_mask=context_mask,
        )

        s = query_x[:, :, 0:1]
        k = query_x[:, :, 1:2]
        t = query_x[:, :, 2:3]
        v = query_x[:, :, 3:4]
        rest = query_x[:, :, 4:]
        log_m, sqrt_t = self.surface_model.engineer_surface_features(s=s, k=k, t=t)

        query_feats = torch.cat([log_m, sqrt_t, v, rest], dim=-1)
        context_features = self.surface_model.cross_attn(
            query_features=query_feats,
            context_memory=context_memory,
            context_mask=context_mask
        )

        log_price_norm = self.surface_model.price_head(
            log_m=log_m,
            sqrt_t=sqrt_t,
            v=v,
            rest=rest,
            context_features=context_features,
        )
        normalized_prices = self.surface_model.decode_log_price_norm(log_price_norm=log_price_norm)
        return normalized_prices * k
