import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import torch
from PIL import Image
from mpl_toolkits.mplot3d.proj3d import transform
from torchvision.transforms import v2

masks = np.load("Inferenced/masks_of_real_scans_from_BW.npy")
image_path = "/media/ali/SecondSSD/MyResearch/Datasets/Bin_by_Bluewrist/BinScans/rgb_depth/ 12-16 /rgb/001101.jpg"
image_path = "/media/ali/SecondSSD/MyResearch/data_generation/new_data_generation_pipeline/output/bin_picking_final/train_pbr/000002/rgb/000015.jpg"

image = np.array(Image.open(image_path)).transpose(2, 0, 1)
# plt.imshow(image.transpose(1, 2, 0))
# plt.show()

tr = v2.Compose([v2.RandomGrayscale(1.0)])
transformed_image = tr(torch.from_numpy(image))

plt.imshow(transformed_image.numpy().transpose(1, 2, 0))
print(transformed_image.numpy().transpose(1, 2, 0)[:10, :10, 0])
plt.show()


# for i, mask in enumerate(masks):
#     sample_mask = mask.transpose(1, 2, 0)
#     sample_mask = np.where(sample_mask > 0.6, 1, 0).astype(np.uint8)
#     segmented_rgb = np.multiply(image, sample_mask)
#
#     plt.imshow(segmented_rgb, cmap="gray")
#     # sns.heatmap(sample_mask[564:987, 138:552, 0])
#     plt.show()

# sample_mask = masks[0].transpose(1, 2, 0)
# sample_mask = np.where(sample_mask > 0.6, 1, 0).astype(np.uint8)
# np.save("/home/ali/Desktop/Mask_MaskRCNN.npy", sample_mask)
# # .imsave("~/Desktop/Mask_MaskRCNN.png", sample_mask[:, :])
