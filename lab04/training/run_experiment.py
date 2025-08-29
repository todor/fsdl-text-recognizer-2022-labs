"""Experiment-running framework."""
import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch

from text_recognizer import lit_models
from training.util import DATA_CLASS_MODULE, import_class, MODEL_CLASS_MODULE, setup_data_and_model_from_args

import inspect


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Minimal Trainer args (PL v2.x)
    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_group.add_argument("--max_epochs", type=int, default=1, help="Number of training epochs")
    trainer_group.add_argument("--accelerator", type=str, default="auto", help="cpu, gpu, tpu, mps, or auto")
    trainer_group.add_argument("--devices", type=str, default=None, help="Number of devices, e.g. 1 or 0,1")
    trainer_group.add_argument("--precision", type=str, default="32", help="Precision: 16, bf16, or 32")
    trainer_group.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit training batches (float=percent, int=num_batches)")
    trainer_group.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit validation batches")
    trainer_group.add_argument("--limit_test_batches", type=float, default=1.0, help="Limit test batches")
    trainer_group.add_argument("--log_every_n_steps", type=int, default=50, help="Logging frequency in steps")
    trainer_group.add_argument("--check_val_every_n_epoch", type=int, default=1, help="Validation frequency in epochs")

    # Backward compatibility for old "--gpus"
    trainer_group.add_argument("--gpus", type=int, default=None, help="Deprecated, use --devices instead")

    # Basic arguments
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--data_class",
        type=str,
        default="MNIST",
        help=f"String identifier for the data class, relative to {DATA_CLASS_MODULE}.",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="MLP",
        help=f"String identifier for the model class, relative to {MODEL_CLASS_MODULE}.",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--stop_early", type=int, default=0)

    # Get the data and model classes so we can add their args
    temp_args, _ = parser.parse_known_args()
    data_class = import_class(f"{DATA_CLASS_MODULE}.{temp_args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{temp_args.model_class}")

    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    lit_model_group = parser.add_argument_group("LitModel Args")
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser




@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=MLP --data_class=MNIST
    ```

    For basic help documentation, run the command
    ```
    python training/run_experiment.py --help
    ```

    The available command line args differ depending on some of the arguments, including --model_class and --data_class.

    To see which command line args are available and read their documentation, provide values for those arguments
    before invoking --help, like so:
    ```
    python training/run_experiment.py --model_class=MLP --data_class=MNIST --help
    """
    parser = _setup_parser()
    args = parser.parse_args()
    data, model = setup_data_and_model_from_args(args)

    lit_model_class = lit_models.BaseLitModel

    # Use TransformerLitModel if the model class or loss indicates a transformer
    transformer_models = ["LineCNNTransformer", "Transformer"]  # add more if needed
    if args.model_class in transformer_models or args.loss == "transformer":
        lit_model_class = lit_models.TransformerLitModel
    
    if args.load_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(args.load_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/cer" if args.loss in ("transformer",) else "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if args.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=args.stop_early
        )
        callbacks.append(early_stopping_callback)

    # Keep only arguments that match Trainer __init__ parameters
    trainer_param_names = inspect.signature(pl.Trainer.__init__).parameters.keys()
    trainer_kwargs = {k: v for k, v in vars(args).items() if k in trainer_param_names}
    
    # Ensure defaults for devices/accelerator
    if torch.cuda.is_available():
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = torch.cuda.device_count()
    else:
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1
    
    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.gpus if args.gpus is not None else args.devices,
        precision=args.precision,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
    )

    # If passing --auto_lr_find, this will set learning rate
    if getattr(args, "auto_lr_find", False):
        lr_finder = trainer.tuner.lr_find(lit_model, datamodule=data)
        new_lr = lr_finder.suggestion()
        lit_model.hparams.lr = new_lr
    
    trainer.fit(lit_model, datamodule=data)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        rank_zero_info(f"Best model saved at: {best_model_path}")
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_model, datamodule=data)


if __name__ == "__main__":
    main()
