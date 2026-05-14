import lightning as L
import torch
from backends.reg_resnet import ResNet50
from backends.seg_maskrcnn import MaskRCNN


class SegLoader(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MaskRCNN(11)
        self.model.eval()

    def forward(self, input):
        with torch.no_grad():
            predictions = self.model(input)

        return predictions


class RegLoader(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ResNet50()
        self.model.eval()

    def forward(self, input):
        with torch.no_grad():
            predictions = self.model(input)

        return predictions