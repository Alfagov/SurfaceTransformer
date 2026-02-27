from datetime import datetime
from pathlib import Path
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import lightning as pl

from models.dataset import SurfaceOptionsDataModule
from models.module import SurfaceTransformerOptionModule, export_surface_model_to_onnx

lr = 3e-4
EPOCHS = 400
LAMBDA_ARB = 10.0

dates_per_batch = 24
context_size = 128
query_size = 128

if __name__ == '__main__':
    torch.manual_seed(42)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checkpoint_dir = Path("./checkpoints") / f"run_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="epoch={epoch:02d}-step={step}",
        save_last=True,
        auto_insert_metric_name=False,
    )

    data_module = SurfaceOptionsDataModule(
        "./data/108105",
        sofr_path="./data/sofr.csv",
        context_size=context_size,
        query_size=query_size,
        batch_size_dates=dates_per_batch,
        num_workers=4,
        seed=42,
    )

    model = SurfaceTransformerOptionModule(
        raw_input_dim=6,
        context_size=context_size,
        query_size=query_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        ff_dim=428,
        head_hidden=428,
        lambda_arb=LAMBDA_ARB,
        theta_floor_base=-0.03,
        theta_floor_slope=0.0393,
        price_log_eps=1e-6,
        log_price_clip_max=20.0,
        learning_rate=lr,
        huber_delta=0.8,
        epochs=EPOCHS
    )

    onnx_path = export_surface_model_to_onnx(
        surface_module=model,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = WandbLogger(project="surfaces", log_model="all")
    logger.watch(model, log_graph=True, log="all")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        inference_mode=False,
        check_val_every_n_epoch=5,
    )

    data_module.setup("train")
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.test_dataloader())
    trainer.test(model, data_module)
