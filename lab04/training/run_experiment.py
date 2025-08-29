"""Experiment-running framework."""
import argparse
from pathlib import Path
import inspect

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only

from text_recognizer import lit_models
from text_recognizer import callbacks as cb
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """Set up argparse with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Trainer args (minimal for PL v2)
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument("--max_epochs", type=int, default=1)
    trainer_group.add_argument("--accelerator", type=str, default="auto")
    trainer_group.add_argument("--devices", type=str, default=None)
    trainer_group.add_argument("--precision", type=str, default="32")
    trainer_group.add_argument("--limit_train_batches", type=float, default=1.0)
    trainer_group.add_argument("--limit_val_batches", type=float, default=1.0)
    trainer_group.add_argument("--limit_test_batches", type=float, default=1.0)
    trainer_group.add_argument("--log_every_n_steps", type=int, default=50)
    trainer_group.add_argument("--check_val_every_n_epoch", type=int, default=1)
    trainer_group.add_argument("--gpus", type=int, default=None)  # backward compatibility

    # Basic experiment args
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--data_class", type=str, default="MNIST",
                        help=f"Data class relative to {DATA_CLASS_MODULE}")
    parser.add_argument("--model_class", type=str, default="MLP",
                        help=f"Model class relative to {MODEL_CLASS_MODULE}")
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--stop_early", type=int, default=0)

    # Get data and model classes to add their custom args
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    # Choose LitModel
    lit_model_class = lit_models.BaseLitModel
    transformer_models = ["LineCNNTransformer", "Transformer"]
    if args.model_class in transformer_models or getattr(args, "loss", None) == "transformer":
        lit_model_class = lit_models.TransformerLitModel

    # Load or instantiate
    if args.load_checkpoint:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    # Logging
    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)

    if args.wandb:
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(model, log_freq=max(100, getattr(args, "log_every_n_steps", 50)))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir
    else:
        logger = pl.loggers.TensorBoardLogger(log_dir)
        experiment_dir = logger.log_dir

    # Checkpoint & summary callbacks
    goldstar_metric = "validation/cer" if getattr(args, "loss", None) == "transformer" else "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    if goldstar_metric == "validation/cer":
        filename_format += "-validation.cer={validation/cer:.3f}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=getattr(args, "check_val_every_n_epoch", 1),
    )
    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback, cb.ModelSizeLogger(), cb.LearningRateMonitor()]
    if args.stop_early:
        callbacks.append(pl.callbacks.EarlyStopping(monitor="validation/loss", mode="min", patience=args.stop_early))
    if args.wandb and getattr(args, "loss", None) == "transformer":
        callbacks.append(cb.ImageToTextLogger())

    # Initialize Trainer
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in inspect.signature(pl.Trainer.__init__).parameters.keys()}

    # Ensure defaults for accelerator/devices
    if torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = torch.cuda.device_count()
    else:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1

    trainer = pl.Trainer(
        max_epochs=getattr(args, "max_epochs", 1),
        accelerator=trainer_kwargs.get("accelerator"),
        devices=getattr(args, "devices") or trainer_kwargs.get("devices"),
        precision=getattr(args, "precision", "32"),
        limit_train_batches=getattr(args, "limit_train_batches", 1.0),
        limit_val_batches=getattr(args, "limit_val_batches", 1.0),
        limit_test_batches=getattr(args, "limit_test_batches", 1.0),
        log_every_n_steps=getattr(args, "log_every_n_steps", 50),
        check_val_every_n_epoch=getattr(args, "check_val_every_n_epoch", 1),
        callbacks=callbacks,
        logger=logger,
    )

    # Auto LR find
    if getattr(args, "auto_lr_find", False):
        lr_finder = trainer.tuner.lr_find(lit_model, datamodule=data)
        lit_model.hparams.lr = lr_finder.suggestion()

    # Train + test
    trainer.fit(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
