# Bin-Picking Pose Estimation

6D object pose estimation system developed for the
[Perception Challenge for Bin-Picking (IBPC 2025)](https://bop.felk.cvut.cz/challenges/),
sponsored by OpenCV.
**Result: 12th overall / 8th among trained methods** on the competition leaderboard.

---

## What this does

Given simultaneous RGB and depth images from a three-camera rig, this system estimates
the full 6D pose (3D position + 3D orientation) of industrial objects in a bin which is
the geometric information a robot arm needs to grasp them reliably.

## Pipeline overview

1. **Instance segmentation** - MaskRCNN detects and isolates each object instance per camera
2. **Epipolar matching** - matches detections across all three cameras using epipolar geometry
3. **Radial distance regression** - ResNet predicts per-pixel distances to object keypoints
4. **RANSAC sphere fitting** - recovers keypoint 3D positions from radial distance votes
5. **Horn's method** - computes initial 6D pose from keypoint correspondences
6. **ICP refinement** - aligns object point cloud to depth data for the final pose estimate

## Attribution

This work builds on the [RCVPose3D methodology](https://github.com/aaronWool/rcvpose3d)
developed in our lab. Contributions in this repo: network backbone selection, full
training infrastructure (PyTorch Lightning, W&B, HPC cluster), three-camera fusion
system, RANSAC improvements, ROS2 service integration,
and training of all models on the competition dataset.

## Repository structure

```
ros2_node/ : ROS2 service node (competition submission entry point)
inference/ : core pose estimation pipeline and utilities
backends/  : segmentation and regression model definitions
training/  : training code for segmentation and regression models
models/    : model download script (weights hosted on HuggingFace)
```

## Model weights

Trained checkpoints are hosted on HuggingFace. Run:

```bash
bash models/download_models.sh
```

*Full README with setup instructions, training details, and results in progress.*
