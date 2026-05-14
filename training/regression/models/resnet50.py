import torch
import torch.nn as nn
from torchvision.models import resnet152, resnet50, resnet18


class RadialDistanceModel(nn.Module):
    def __init__(self, num_keypoints=4):
        super(RadialDistanceModel, self).__init__()
        self.resnet50 = resnet50(weights="DEFAULT")
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        self.regressor = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, num_keypoints, kernel_size=3, padding=1)
        )


    def forward(self, x):
        # Converting the batch to pytorch tensor
        im_size = x.size()
        x = self.resnet50(x)
        x = self.regressor(x)
        x = nn.functional.interpolate(x, size=(im_size[2], im_size[3]), mode='bilinear',
                                                     align_corners=False)


        return x

if __name__ == "__main__":
    model = RadialDistanceModel().to(device="cuda")
    x = torch.randn(2, 3, 1200, 1920).to(device="cuda")
    tmp_x = model.resnet50(x)
    print(model)
    print(tmp_x.shape)
    exit()
    torch.cuda.reset_peak_memory_stats()
    output = model(x)

    vram_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"Allocated: {vram_allocated:.2f} MB")
    print(f"Reserved: {vram_reserved:.2f} MB")
    print(f"Peak Allocated: {vram_peak:.2f} MB")
