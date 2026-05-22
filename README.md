# Bin-Picking Pose Estimation

6D object pose estimation system developed for the
[Perception Challenge for Bin-Picking (IBPC 2025)](https://bpc.opencv.org/),
sponsored by OpenCV.

**Result: 12th overall / 8th among trained methods** on the competition leaderboard.

---

## What this does

Given simultaneous RGB and depth images from a calibrated three-camera rig, the system
estimates the full 6D pose — 3D position and 3D orientation — of industrial objects
inside a bin. This is the geometric information a robot arm needs to reliably plan and
execute a grasp.
---

## Pipeline

```
3 RGB-D cameras
      │
      ▼
Instance Segmentation (MaskRCNN)
Per-camera object detection and mask generation
      │
      ▼
Epipolar Matching
Cross-camera detection matching using epipolar geometry
to establish consistent object identities across views
      │
      ▼
Radial Distance Regression (ResNet)
Per-pixel prediction of radial distances to predefined
object keypoints, producing a dense distance map per object
      │
      ▼
RANSAC Sphere Fitting
3D keypoint localisation by fitting spheres to the
merged multi-view point cloud via adaptive RANSAC
      │
      ▼
Horn's Method
Closed-form 6D pose estimation from keypoint correspondences
      │
      ▼
ICP Refinement
Point cloud alignment to depth observations for final pose
```

---

## Repository structure

```
ros2_node/    — ROS2 service node (competition submission entry point)
inference/    — pose estimation pipeline and utilities
  utils/      — Horn solver, RANSAC, epipolar matching, ICP, model loaders
backends/     — MaskRCNN and ResNet model definitions
training/
  segmentation/ — MaskRCNN training code
  regression/   — radial distance regression training code
models/       — model weight download script
```

---

## Models

One MaskRCNN segmentation model and ten per-object ResNet regression models were
trained on the IPD dataset. Weights are hosted on
[HuggingFace](https://huggingface.co/Ali-Rafiaei/ibpc-pose-estimator):

```bash
pip install huggingface_hub
python models/download_models.py
```

The node expects checkpoints at:

```
<model_dir>/
  checkpoints/
    segmentation/
      all_objs.ckpt
    regression/
      obj_<id>.ckpt
```

---

## Technical notes

**Multi-camera fusion:** RANSAC sphere fitting operates on the merged point cloud
from all three cameras rather than per-camera, improving keypoint localisation
robustness under occlusion.

**Per-object regression:** A separate ResNet regression model was trained for each
of the 10 competition objects, with one shared MaskRCNN handling detection and
segmentation across all objects.

**Mixed precision training:** Regression models were trained with `16-mixed`
precision via PyTorch Lightning, reducing GPU memory usage during training on
the HPC cluster.

---

## Competition context

The IBPC competition provided a ROS2 framework and Docker environment that handled
camera data delivery and the service interface. The system runs inside that
environment and is not designed to execute standalone.

**This repository contains the following:**
- Full pose estimation pipeline
- Epipolar multi-view matching
- Adaptive RANSAC keypoint localisation (Numba JIT-compiled)
- Horn's method and ICP integration
- Both training pipelines (segmentation and regression)
- All trained model weights

The ROS2 node structure, service interface, and `Camera` class were provided
by the competition organizers.

---

## Attribution

Builds on the [RCVPose methodology](https://github.com/aaronWool/rcvpose3d) developed
in our lab, specifically the radial distance regression approach to keypoint
localisation. My Contributions: backbone selection, full training infrastructure
(Using PyTorch Lightning, and W&B, on our HPC cluster), three-camera fusion,
adaptive RANSAC improvements, and ROS2 integration.

---

## Leaderboard

| | Result |
|---|---|
| Overall rank | 12th / mixed leaderboard (one-shot + trained methods) |
| Rank among trained methods | 8th |
| Competition | IBPC 2025 — Perception Challenge for Bin-Picking (OpenCV) |
