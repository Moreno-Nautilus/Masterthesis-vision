# Multi-View Geometric Perception Baseline

This repository implements a **multi-view geometric perception pipeline** for detecting and estimating the pose of objects using **two ZED cameras**.

The current system is a **depth-based baseline** that fuses point clouds from multiple cameras, removes planar surfaces (such as a table), segments the scene into clusters, and aligns a **CAD model** to the observed geometry using **RANSAC + ICP**.

This pipeline serves as the **baseline before integrating learned perception methods** such as **SAM, DINOv2, and FoundationPose**.

---

# Pipeline Overview

The baseline pipeline processes synchronized depth images from two cameras and estimates a **6D pose of the target object**.

```
Depth Images (ZED1 + ZED2)
        │
        ▼
Backprojection (Depth → Point Cloud)
        │
        ▼
Transform to Base Frame
        │
        ▼
Multi-View Fusion
        │
        ▼
Plane Removal (RANSAC)
        │
        ▼
Height Filtering
        │
        ▼
Workspace Cropping
        │
        ▼
DBSCAN Clustering
        │
        ▼
Cluster Refinement
        │
        ▼
CAD Alignment (RANSAC + ICP)
        │
        ▼
6D Object Pose (T_base_obj)
```

The resulting pose estimate can later be used for robotic manipulation.

---

# Running the Pipeline

The pipeline can be started with:

```
python -m src.perception.ros.run_multiview_ros
```

This script:

1. Subscribes to ZED camera depth topics
2. Synchronizes the incoming frames
3. Converts depth images into point clouds
4. Transforms point clouds into the robot base frame
5. Fuses multiple camera views
6. Runs the geometric perception pipeline
7. Outputs pose estimates and debugging visualizations

---

# Repository Structure

```
src/perception
│
├── ros
│   ├── run_multiview_ros.py
│   └── multicam_grabber.py
│
├── pipeline.py
├── pipeline_multiview.py
│
├── fusion.py
├── backproject.py
│
├── segmentation.py
├── pose_icp.py
│
├── view.py
├── se3.py
└── io_extrinsics.py
```

---

# Core Components

## run_multiview_ros.py

Main entry point of the perception system.

Responsibilities:

- Load CAD models
- Load camera extrinsics
- Initialize the perception pipeline
- Subscribe to ROS camera streams
- Trigger pipeline execution
- Save debugging visualizations

Primary output:

```
T_base_obj
```

This represents the object pose in the robot base frame.

---

# ROS Interface

## multicam_grabber.py

Handles all ROS camera communication.

Subscribes to:

- ZED depth images
- Camera intrinsics (`CameraInfo`)

Produces synchronized **View objects** containing:

```
View(
    depth,
    camera intrinsics,
    camera extrinsics
)
```

These objects are passed into the perception pipeline.

---

# Data Structures

## view.py

Defines the **View** object used throughout the pipeline.

```
View
 ├── rgb
 ├── depth
 ├── K
 └── T_base_cam
```

Currently RGB is unused because the baseline is purely geometric.

---

# Depth Processing

## backproject.py

Converts depth images into 3D point clouds using the pinhole camera model.

Projection equations:

```
x = (u − cx) * z / fx
y = (v − cy) * z / fy
z = depth
```

Output:

```
Nx3 array of 3D points in camera frame
```

---

# Multi-View Fusion

## fusion.py

Combines point clouds from all cameras.

Steps:

1. Backproject depth into camera point clouds
2. Transform each point cloud into the base frame
3. Merge all points
4. Downsample using voxel filtering

Output:

```
fused_point_cloud_base
```

---

# Scene Segmentation

## segmentation.py

Separates objects from the environment using geometric operations.

### Plane Removal

Detects the dominant planar surface (table) using **RANSAC**.

```
points
├── plane points
└── remaining scene points
```

### Height Filtering

Keeps only points within a certain distance above the plane.

### Workspace Cropping

Restricts points to a circular workspace region.

### Clustering

Uses **DBSCAN** to segment remaining points into candidate objects.

---

# Pose Estimation

## pose_icp.py

Aligns the observed cluster with the CAD model.

Procedure:

1. Downsample observed cluster
2. Downsample CAD model
3. Compute FPFH features
4. RANSAC feature matching
5. ICP refinement

Output:

```
T_base_obj
```

which represents the estimated object pose.

---

# Multi-View Pipeline Wrapper

## pipeline_multiview.py

Connects multi-camera fusion with the single-scene perception pipeline.

Flow:

```
Views
  → fuse_views_to_points_base()
  → pipeline.run()
```

---

# Core Pipeline

## pipeline.py

Implements the main geometric perception algorithm.

### Stage 1 — Plane Removal

Detect the dominant planar surface using RANSAC.

### Stage 2 — Height Filtering

Keep points within a band above the plane.

### Stage 3 — Workspace Cropping

Restrict points to a circular workspace region.

### Stage 4 — Clustering

Segment objects using DBSCAN.

### Stage 5 — Cluster Refinement

If the largest cluster still contains planar geometry:

- remove plane again
- recluster

### Stage 6 — Pose Estimation

For each cluster:

```
CAD model
      vs
observed cluster
```

Run ICP alignment and compute alignment metrics.

The cluster with the best score is selected as the detected object.

---

# Coordinate Frames

Transform notation:

```
T_A_B
```

Transforms points from frame **B → A**.

Example:

```
T_base_cam
```

Transforms camera points into the robot base frame.

Object pose:

```
T_base_obj
```

---

# Debug Visualizations

The system generates Plotly HTML visualizations.

### multiview_above_plane_debug.html

Shows:

- fused point cloud
- detected plane
- points above plane

### multiview_detection_debug.html

Shows:

- clusters
- CAD alignment
- detected objects

### multiview_workspace_with_icp.html

Shows:

- fused workspace
- final pose estimate

---

# Current Limitations

The baseline is purely geometric.

Limitations include:

- no semantic understanding
- clustering may merge objects
- ICP may lock onto wrong structures
- sensitivity to clutter and occlusions

---

# Planned Improvements

The next version of the pipeline will integrate **foundation models**.

Future architecture:

```
RGB-D
  │
  ▼
SAM (segmentation)
  │
  ▼
DINOv2 (semantic identification)
  │
  ▼
FoundationPose (6D pose estimation)
  │
  ▼
Pose arbitration across cameras
  │
  ▼
Franka wrist camera refinement
  │
  ▼
Grasp execution
```

---

# Goal

The final system aims to provide a **robust multi-view perception pipeline for robotic manipulation**, capable of:

- reliable 6D pose estimation
- multi-object scenes
- wrist-camera refinement
- autonomous grasp execution