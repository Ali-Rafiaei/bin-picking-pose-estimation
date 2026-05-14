import lightning as L
from torch.optim import lr_scheduler
from torch.optim import Adam, SGD
from models.models import ResNet18, ResNet50, ResNet152
import torch


class RadialDistanceTrainer(L.LightningModule):
    def __init__(self, args, num_keypoints=4):
        super().__init__()
        self.learning_rate = args.learning_rate
        self.args = args
        if isinstance(args.objs_to_use, int):
            # self.model = ResNet50(num_keypoints, args.input_include_depth)
            self.model = ResNet50(num_keypoints, args.input_include_depth)
        else:
            self.model = ResNet50(num_keypoints, args.input_include_depth)

        self.loss_radial = torch.nn.L1Loss(reduction='sum')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, radial_maps = batch
        preds = self(images)
        loss = self.loss_radial(preds[torch.where(radial_maps != 0)],
                                radial_maps[torch.where(radial_maps != 0)]) / float(len(torch.nonzero(radial_maps)))

        model_accuracy = float(torch.sum(
            torch.where(torch.abs(preds - radial_maps)[torch.where(radial_maps != 0)] <= 5, 1, 0)) / float(
            len(torch.nonzero(radial_maps))))

        # self.log("train_radial_loss", loss, sync_dist=True, batch_size=self.args.batch_size)
        # self.log('train_radial_accuracy', model_accuracy, sync_dist=True, batch_size=self.args.batch_size)
        self.log("train_radial_loss", loss, sync_dist=True, prog_bar=True)
        self.log('train_radial_accuracy', model_accuracy, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, radial_maps = batch
        preds = self(images)
        loss = self.loss_radial(preds[torch.where(radial_maps != 0)],
                                radial_maps[torch.where(radial_maps != 0)]) / float(len(torch.nonzero(radial_maps)))

        model_accuracy = float(torch.sum(
            torch.where(torch.abs(preds - radial_maps)[torch.where(radial_maps != 0)] <= 5, 1, 0)) / float(
            len(torch.nonzero(radial_maps))))

        # self.log("val_radial_loss", loss, sync_dist=True, batch_size=self.args.batch_size)
        # self.log('val_radial_accuracy', model_accuracy, sync_dist=True, batch_size=self.args.batch_size)

        self.log("val_radial_loss", loss, sync_dist=True, prog_bar=True)
        self.log('val_radial_accuracy', model_accuracy, sync_dist=True)

        return loss

    def configure_optimizers(self):
        # optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # optimizer = (self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)/
        optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)

        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update LR every batch step
                "frequency": 1,
            }
        }
