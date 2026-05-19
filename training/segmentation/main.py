import os
from train import MaskRCNNTrainer
from data_loader import IpdDataset, collate_fn
from lightning.pytorch import Trainer, callbacks
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pbr_path",
                        type=str,
                        default=None)

    parser.add_argument("--optim",
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])

    parser.add_argument("--batch_size",
                        type=int,
                        default=1)

    parser.add_argument("--initial_lr",
                        type=float,
                        default=1e-4)

    parser.add_argument('--max_epochs',
                        type=int,
                        default=20000)


    args = parser.parse_args()

    checkpoint_callback = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_total_loss",
        mode="min",
        filename="Best-Val-loss-{epoch:02d}-{val_radial_loss:.2f}",
        save_last=True,
        enable_version_counter=True,
        every_n_epochs=1
    )

    checkpoint_callback_on_train_loss = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="train_total_loss",
        mode="min",
        filename="Best-train-loss-{epoch:02d}-{step}-{train_radial_loss:.2f}",
        save_last=True,
        enable_version_counter=True,
        every_n_train_steps=20
    )

    train_dataset = IpdDataset(args,
                               mode='train')
    val_dataset = IpdDataset(args,
                             mode='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=1)

    model = MaskRCNNTrainer(args)

    wandb_logger = WandbLogger(project="Test")
    wandb_logger.watch(model)

    gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    gpu_devices = [int(device) for device in gpu_devices.split(",") if device.isdigit()]
    number_of_gpus = len(gpu_devices)

    trainer = Trainer(max_epochs=20000, devices=[0, 1], accelerator='gpu',
                      check_val_every_n_epoch=1,
                      strategy='ddp', log_every_n_steps=8,
                      callbacks=[checkpoint_callback, checkpoint_callback_on_train_loss], logger=wandb_logger)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
