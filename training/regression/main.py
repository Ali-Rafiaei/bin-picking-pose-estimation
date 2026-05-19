import os
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import PyTorchProfiler
from torch.utils.data import DataLoader

from train import RadialDistanceTrainer
from data_loader import RadialMapLoader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pbr_path",
                        type=str,
                        default=None)

    parser.add_argument("--batch_size",
                        type=int,
                        default=2)

    parser.add_argument("--gradient_accumulation_value",
                        type=int,
                        default=8)

    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4)

    parser.add_argument("--precision",
                        type=str,
                        default="16-mixed")

    parser.add_argument('--max_epochs',
                        type=int,
                        default=20000)

    parser.add_argument("--split_path",
                        type=str,
                        default="dataset_splits")

    parser.add_argument("--input_include_depth",
                        type=bool,
                        default=False)

    parser.add_argument("--objs_to_use",
                        default=20)

    parser.add_argument("--kpoint_type",
                        type=str,
                        default="keygnet")

    args = parser.parse_args()

    checkpoint_callback = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_radial_loss",
        mode="min",
        filename="Best-Val-loss-{epoch:02d}-{val_radial_loss:.2f}",
        save_last=True,
        enable_version_counter=True,
        every_n_epochs=1
    )

    checkpoint_callback_on_train_loss = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="train_radial_loss",
        mode="min",
        filename="Best-train-loss-{epoch:02d}-{step}-{train_radial_loss:.2f}",
        enable_version_counter=True,
        every_n_train_steps=20
    )

    model = RadialDistanceTrainer(args)

    train_loader = DataLoader(RadialMapLoader(args, set="train"),
                              batch_size=int(args.batch_size),
                              shuffle=True,
                              num_workers=1)

    val_loader = DataLoader(RadialMapLoader(args, set="val"),
                            batch_size=int(args.batch_size),
                            shuffle=False,
                            num_workers=1)

    wandb_logger = WandbLogger(project="Test", config=args)
    wandb_logger.watch(model)


    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_devices = [int(device) for device in gpu_devices.split(",") if device.isdigit()]
    number_of_gpus = len(gpu_devices)

    trainer = Trainer(max_epochs=args.max_epochs,
                      accelerator='gpu', check_val_every_n_epoch=1,
                      log_every_n_steps=16,
                      callbacks=[checkpoint_callback, checkpoint_callback_on_train_loss], logger=wandb_logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)