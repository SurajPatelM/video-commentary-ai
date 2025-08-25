import os
import optuna
from omegaconf import OmegaConf
from lightning import Trainer as LTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from data.datamodule import VideoDataModule
from models.train_lightning import TrainerModule
import torch
import hydra
from hydra import initialize, compose

# Objective function for Optuna
def objective(trial):
    # Suggest a learning rate
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

    # Load the base config using Hydra
    with initialize(config_path="configs"):
        cfg = compose(config_name="default")

    # Override the learning rate and training epochs
    cfg.trainer.lr = lr
    cfg.trainer.epochs = 40
    cfg.trainer.exp_name = f"optuna_trial_{trial.number}"
    cfg.trainer.output_dir = "./optuna_checkpoints"

    # Optional: Clear MPS cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # DataModule
    datamodule = VideoDataModule(cfg)
    datamodule.setup(stage="fit")

    # Model
    model = TrainerModule(cfg)

    # Logger
    logger = TensorBoardLogger(
        save_dir=f"./tensorboard_logs/{cfg.trainer.exp_name}",
        name="video_captioning",
    )

    # Lightning Trainer
    trainer = LTrainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.epochs,
        precision=cfg.trainer.precision,
        default_root_dir=cfg.trainer.output_dir,
        logger=logger,
        enable_progress_bar=False,  # Optional: disable progress bar for cleaner output
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Retrieve metric — assumes val_metric is logged during validation
    best_val_score = trainer.callback_metrics.get("val_metric")

    # Convert to float
    if best_val_score is None:
        return float("inf")  # or 0.0 if you're maximizing

    return best_val_score.item()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="lr_tuning")
    study.optimize(objective, n_trials=12)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
