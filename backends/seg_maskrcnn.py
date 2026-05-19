import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.base_model = maskrcnn_resnet50_fpn_v2(weights=None)

        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)


    def forward(self, images, targets=None):
            return self.base_model(images, targets)


if __name__ == "__main__":
    model = MaskRCNN(num_classes=6)
    images = torch.rand(2, 3, 1920, 1200)
    targets = [{"boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), "labels": torch.tensor([1, 1]),
            "masks": torch.rand((2, 1920, 1200))},
               {"boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), "labels": torch.tensor([1, 1]),
                "masks": torch.rand((2, 1920, 1200))}]
    print(maskrcnn_resnet50_fpn_v2(weights="DEFAULT"))