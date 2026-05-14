import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Load pre-trained Mask R-CNN
        self.base_model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

        # Modify the classifier to match the number of classes
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Modify the mask predictor
        in_features_mask = self.base_model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.base_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)


    def forward(self, images, targets=None):
            return self.base_model(images, targets)


if __name__ == "__main__":
    # Example usage
    model = MaskRCNN(num_classes=6)
    images = torch.rand(2, 3, 1920, 1200)
    # targets = [{"boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), "labels": torch.tensor([1, 1]),
    #             "masks": torch.rand((2, 1920, 1200)), "visibility_ratio": torch.tensor([0.9, 0.8])}]
    targets = [{"boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), "labels": torch.tensor([1, 1]),
            "masks": torch.rand((2, 1920, 1200))},
               {"boxes": torch.tensor([[0, 0, 100, 100], [50, 50, 150, 150]]), "labels": torch.tensor([1, 1]),
                "masks": torch.rand((2, 1920, 1200))}]

    # for k, v in model(images, targets).items():
    #     print(k, " : ", v)
    # losses = model(images, targets)
    # print(losses.)
    # print(sum([loss for loss in losses.values()]))
    # print(sum(losses.values()))
    print(maskrcnn_resnet50_fpn_v2(weights="DEFAULT"))