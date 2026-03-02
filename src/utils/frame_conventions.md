# Frame Conventions – 6D Pose & Multi-View Fusion

This document defines all coordinate frame conventions and transform semantics used in the grasp perception pipeline.
These conventions must remain consistent across perception, fusion, calibration, and grasping.

---

## 1. Transform Notation

We use the convention:

> T_A_B maps points expressed in frame B into frame A.

If a point is expressed in frame B:

p_A = T_A_B ⊕ p_B

In code:

```python
p_A = T_A_B.transform_points(p_B)
```

Matrix form:

```
T_A_B = [ R  t ]
        [ 0  1 ]
```

Where:

- R = rotation from B to A  
- t = translation of origin of B expressed in A  

---

## 2. Core Frames

We define the following frames:

### cam
Camera optical frame.

- Origin: camera optical center  
- Axes: OpenCV convention (Z forward, X right, Y down)  
- Depth backprojection produces points in this frame  

---

### base
Robot base frame.

- Global reference frame in real-world operation  
- All fused point clouds should be expressed in this frame  
- All grasp poses must be expressed in this frame  

This is the primary global frame in real experiments.

---

### obj
Object frame.

- Defined by CAD model  
- If CAD is centered, origin = CAD centroid  
- Pose estimation solves for T_cam_obj  

---

### world (simulation only)
Used only in simulation.

In real-world experiments, base replaces world as global frame.

---

## 3. Perception Output Convention

The 6D pose estimator must output:

T_cam_obj

Meaning:

Pose of object in camera frame.

Applying it:

```python
P_cam_pred = T_cam_obj.transform_points(P_obj)
```

This aligns CAD model points to observed camera points.

---

## 4. Transform Chaining

### Camera → Base

After calibration we have:

T_base_cam

Then:

```python
T_base_obj = T_base_cam @ T_cam_obj
```

---

### Grasp Mapping

If grasp is defined in object frame:

```python
T_base_grasp = T_base_obj @ T_obj_grasp
```

---

## 5. Multi-View Fusion Convention

For each view i:

- Depth is backprojected → P_cam_i  
- Convert to base frame:

```python
P_base_i = T_base_cam_i.transform_points(P_cam_i)
```

All views are fused in base frame.

---

## 6. CAD Model Convention

CAD point clouds are centered (center=True).

Therefore:

- Object frame origin = CAD centroid  
- Grasp definitions must assume this origin  

---

## 7. SE3 Composition Rule

We use:

```python
T3 = T1 @ T2
```

Which means:

```python
T3 = T1.compose(T2)
```

Order matters.

If:

```python
T_base_obj = T_base_cam @ T_cam_obj
```

Then points transform as:

p_base = T_base_obj ⊕ p_obj

---

## 8. Golden Rule

Every transform name must encode source and target frames explicitly.

Correct:
- T_base_cam
- T_cam_obj
- T_base_obj

Incorrect:
- T_world
- T_obs
- T_model

Ambiguity in naming causes silent pose errors.