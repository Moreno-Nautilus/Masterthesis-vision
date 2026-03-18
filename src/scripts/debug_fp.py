"""
Standalone FoundationPose diagnostic - run OUTSIDE of ROS2.
Save a debug frame first, then run this script.

Usage:
  python diagnose_fp.py --frame-dir outputs/foundationpose/debug_frames/frame_000001
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import cv2

def diagnose(frame_dir: str, mesh_path: str, mesh_scale: float = 1.0):
    """
    Standalone FP diagnosis - no ROS, no wrappers.
    """
    frame_dir = Path(frame_dir)
    
    # Add FP to path
    FP_ROOT = Path("external/FoundationPose").resolve()
    if not FP_ROOT.exists():
        print(f"ERROR: FoundationPose not found at {FP_ROOT}")
        return
    sys.path.insert(0, str(FP_ROOT))
    
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    
    # Load inputs
    rgb_path = frame_dir / "rgb.png"
    depth_path = frame_dir / "depth.npy"
    mask_path = frame_dir / "mask.png"
    K_path = frame_dir / "K.npy"
    
    for p in [rgb_path, depth_path, mask_path, K_path]:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            return
    
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    depth = np.load(str(depth_path)).astype(np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    K = np.load(str(K_path)).astype(np.float32)
    
    print("=" * 70)
    print("INPUT DIAGNOSTICS")
    print("=" * 70)
    
    # RGB check
    print(f"RGB: shape={rgb.shape} dtype={rgb.dtype} range=[{rgb.min()}, {rgb.max()}]")
    
    # Depth check - CRITICAL
    depth_in_mask = depth[mask]
    valid_depth = depth_in_mask[(depth_in_mask > 0) & np.isfinite(depth_in_mask)]
    print(f"\nDepth: shape={depth.shape} dtype={depth.dtype}")
    print(f"  Full image: min={np.nanmin(depth):.4f} max={np.nanmax(depth):.4f}")
    if len(valid_depth) > 0:
        print(f"  In mask: min={valid_depth.min():.4f} median={np.median(valid_depth):.4f} max={valid_depth.max():.4f}")
        print(f"  Valid pixels in mask: {len(valid_depth)} / {mask.sum()}")
        
        # Sanity check depth units
        med = np.median(valid_depth)
        if med > 10:
            print("  ⚠️  WARNING: Depth median > 10 - likely in MILLIMETERS, not meters!")
        elif med < 0.1:
            print("  ⚠️  WARNING: Depth median < 0.1m - object very close or units wrong")
        else:
            print(f"  ✓ Depth looks reasonable for object at ~{med:.2f}m")
    else:
        print("  ⚠️  ERROR: No valid depth in mask!")
        return
    
    # Mask check
    print(f"\nMask: shape={mask.shape} pixels={mask.sum()}")
    ys, xs = np.where(mask)
    bbox = (xs.min(), ys.min(), xs.max(), ys.max())
    print(f"  BBox: x=[{bbox[0]}, {bbox[2]}] y=[{bbox[1]}, {bbox[3]}]")
    print(f"  Size: {bbox[2]-bbox[0]} x {bbox[3]-bbox[1]} px")
    
    # Intrinsics check
    print(f"\nIntrinsics K:")
    print(f"  fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    
    # Mesh check - CRITICAL
    print("\n" + "=" * 70)
    print("MESH DIAGNOSTICS")
    print("=" * 70)
    
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        print(f"ERROR: Mesh not found at {mesh_path}")
        return
    
    mesh = trimesh.load(str(mesh_path))
    print(f"Original mesh: {mesh_path.name}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  extents (raw): [{mesh.extents[0]:.4f}, {mesh.extents[1]:.4f}, {mesh.extents[2]:.4f}]")
    
    # Detect units
    max_extent = mesh.extents.max()
    if max_extent > 10:
        print(f"  → Mesh appears to be in MILLIMETERS (max extent = {max_extent:.1f})")
        recommended_scale = 0.001
    elif max_extent > 1:
        print(f"  → Mesh appears to be in CENTIMETERS (max extent = {max_extent:.2f})")
        recommended_scale = 0.01
    else:
        print(f"  → Mesh appears to be in METERS (max extent = {max_extent:.4f})")
        recommended_scale = 1.0
    
    print(f"  Recommended mesh_scale: {recommended_scale}")
    print(f"  You're using mesh_scale: {mesh_scale}")
    
    # Apply scale
    mesh.apply_scale(mesh_scale)
    print(f"\nAfter applying mesh_scale={mesh_scale}:")
    print(f"  extents: [{mesh.extents[0]:.4f}, {mesh.extents[1]:.4f}, {mesh.extents[2]:.4f}] meters")
    
    max_scaled = mesh.extents.max()
    if max_scaled < 0.01:
        print(f"  ⚠️  CRITICAL: Mesh is {max_scaled*1000:.2f}mm - WAY TOO SMALL!")
        print(f"      This will cause FP to fail or give nonsense poses.")
        if mesh_scale == 0.01 and recommended_scale == 0.001:
            print(f"      Your STL is in mm but you're using scale=0.01 (for cm)")
            print(f"      → Try mesh_scale=0.001 instead")
    elif max_scaled > 1.0:
        print(f"  ⚠️  WARNING: Mesh is {max_scaled:.2f}m - quite large")
    else:
        print(f"  ✓ Mesh size looks reasonable: ~{max_scaled*100:.1f}cm")
    
    # Force vertex normals
    _ = mesh.vertex_normals
    
    # Build estimator
    print("\n" + "=" * 70)
    print("FOUNDATIONPOSE INITIALIZATION")
    print("=" * 70)
    
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    
    debug_dir = Path("outputs/fp_diagnosis")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    est = FoundationPose(
        model_pts=np.asarray(mesh.vertices, dtype=np.float32),
        model_normals=np.asarray(mesh.vertex_normals, dtype=np.float32),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=3,  # Max debug
        glctx=glctx,
    )
    print("✓ FoundationPose estimator created")
    
    # Run registration
    print("\n" + "=" * 70)
    print("RUNNING REGISTRATION")
    print("=" * 70)
    
    pose = est.register(
        K=K,
        rgb=rgb.astype(np.uint8),
        depth=depth,
        ob_mask=mask,
        iteration=5,
    )
    
    if pose is None:
        print("❌ register() returned None!")
        return
    
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    
    print("\n" + "=" * 70)
    print("POSE ANALYSIS")
    print("=" * 70)
    
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    print(f"Translation t: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
    print(f"  |t| = {np.linalg.norm(t):.4f}m")
    
    # Rotation analysis
    trace = np.trace(R)
    det = np.linalg.det(R)
    print(f"\nRotation R:")
    print(f"  trace(R) = {trace:.4f}")
    print(f"  det(R) = {det:.6f}")
    
    # Check orthogonality
    RRT = R @ R.T
    ortho_err = np.abs(RRT - np.eye(3)).max()
    print(f"  max|R@R.T - I| = {ortho_err:.6f}")
    if ortho_err > 0.01:
        print("  ⚠️  WARNING: R is not orthogonal!")
    
    # Compute angle from trace: trace(R) = 1 + 2*cos(theta)
    cos_theta = np.clip((trace - 1) / 2, -1, 1)
    angle_deg = np.degrees(np.arccos(cos_theta))
    print(f"  Rotation angle from identity: {angle_deg:.1f}°")
    
    # Orientation check
    if trace < 0.5:
        print(f"  ⚠️  WARNING: trace < 0.5 suggests flipped/inverted orientation")
    
    # Z-axis analysis
    print(f"\nZ-axis analysis:")
    print(f"  Object Z in camera frame: {t[2]:.4f}m")
    print(f"  Expected (from depth): ~{np.median(valid_depth):.4f}m")
    
    z_error = abs(t[2] - np.median(valid_depth))
    if t[2] < 0:
        print(f"  ❌ NEGATIVE Z - object behind camera!")
        print(f"     This means the pose is inverted. Try np.linalg.inv(pose)")
    elif z_error > 0.1:
        print(f"  ⚠️  Z differs from depth by {z_error:.3f}m")
    else:
        print(f"  ✓ Z matches depth within {z_error:.3f}m")
    
    # Check inverse
    print(f"\nInverse pose check:")
    pose_inv = np.linalg.inv(pose)
    t_inv = pose_inv[:3, 3]
    print(f"  inv(pose) translation: [{t_inv[0]:.4f}, {t_inv[1]:.4f}, {t_inv[2]:.4f}]")
    
    # Which one is correct?
    if t[2] > 0 and 0.3 < t[2] < 1.0:
        print(f"\n✓ RAW pose looks correct (Z={t[2]:.3f}m in valid range)")
    elif t_inv[2] > 0 and 0.3 < t_inv[2] < 1.0:
        print(f"\n✓ INVERTED pose looks correct (Z={t_inv[2]:.3f}m in valid range)")
        print(f"  → Use np.linalg.inv(pose) as your output")
    else:
        print(f"\n⚠️  Neither pose has Z in expected range [0.3, 1.0]m")
        print(f"     Raw Z={t[2]:.3f}m, Inv Z={t_inv[2]:.3f}m")
    
    # Save visualization
    print(f"\n" + "=" * 70)
    print("DEBUG OUTPUT")
    print("=" * 70)
    print(f"Check {debug_dir} for FP internal debug images")
    
    return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-dir", required=True, help="Path to debug frame directory")
    parser.add_argument("--mesh", default="Data/CAD_Models/cooling_base.stl")
    parser.add_argument("--mesh-scale", type=float, default=1.0, 
                        help="Mesh scale factor (1.0=meters, 0.001=mm->m, 0.01=cm->m)")
    args = parser.parse_args()
    
    diagnose(args.frame_dir, args.mesh, args.mesh_scale)