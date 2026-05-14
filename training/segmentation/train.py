import torch
import lightning as L
from torch.optim import Adam
from model import MaskRCNN


class MaskRCNNTrainer(L.LightningModule):
    def __init__(self, args, num_classes=6):
        super().__init__()
        self.model = MaskRCNN(num_classes)
        self.learning_rate = args.initial_lr
        self.args = args
        self.save_hyperparameters()

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        losses = self.model(images, targets)
        total_loss = sum(losses.values())

        # Log individual losses
        for k, v in losses.items():
            self.log(f"train_{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                     batch_size=self.args.batch_size)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=self.args.batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass but with no gradient calculation
        self.model.train()
        with torch.no_grad():
            losses = self.model(images, targets)

        total_loss = sum(losses.values())

        # Log individual losses
        for k, v in losses.items():
            self.log(f"val_{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                     batch_size=self.args.batch_size)
        self.log("val_total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                 batch_size=self.args.batch_size)

        return total_loss

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.learning_rate)
