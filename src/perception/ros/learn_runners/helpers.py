import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Header

# Geodesic angle (deg) between two rotation matrices.
def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return np.degrees((Rotation.from_matrix(R1).inv() * Rotation.from_matrix(R2)).magnitude())

# Rotation matrix → quaternion [x,y,z,w] (branch on the largest diagonal term).
def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat().astype(np.float32)

# Build a ROS Pose from a translation + quaternion.
def quaternion_xyzw_to_pose_msg(t_xyz: np.ndarray, q_xyzw: np.ndarray) -> Pose:
    msg = Pose()
    msg.position.x = float(t_xyz[0])
    msg.position.y = float(t_xyz[1])
    msg.position.z = float(t_xyz[2])
    msg.orientation.x = float(q_xyzw[0])
    msg.orientation.y = float(q_xyzw[1])
    msg.orientation.z = float(q_xyzw[2])
    msg.orientation.w = float(q_xyzw[3])
    return msg


def T_to_pose_msg(T: np.ndarray) -> Pose:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    t = T[:3, 3]
    q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])
    return quaternion_xyzw_to_pose_msg(t, q)


def T_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.pose = T_to_pose_msg(T)
    return msg

def bbox_size_xyxy(b: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = b
    return x1 - x0, y1 - y0

def _dprint(*args, debug: bool = False, **kwargs) -> None:
    # Cheap global gate for timing/debug prints outside ROS logging.
    if debug:
        # flush so this interleaves deterministically with _UnifiedLogger output
        # (both are on stdout now) when the run is redirected to a single file.
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

def save_init_pose_render(
    T_base: np.ndarray,
    model_pcd,
    obj_id: str,
    save_path: str,
    accepted: bool,
    gt_pos=None,
) -> None:
    # Save a 3D scatter PNG of the model cloud at the estimated pose (init debug viz).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Transform model cloud to base frame
        pts = np.asarray(model_pcd.points)
        R, t = T_base[:3, :3], T_base[:3, 3]
        pts_base = (R @ pts.T).T + t

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        color = '#3366CC' if accepted else "#AF2323"
        ax.scatter(pts_base[::3, 0], pts_base[::3, 1], pts_base[::3, 2],
                   s=1, c=color, alpha=0.4)

        # Draw pose axes
        for axis_i, axis_color in enumerate(['r', 'g', 'b']):
            axis_end = t + R[:, axis_i] * 0.05
            ax.plot([t[0], axis_end[0]], [t[1], axis_end[1]], [t[2], axis_end[2]],
                    color=axis_color, linewidth=2)

        if gt_pos is not None:
            ax.scatter(*gt_pos, s=80, c='lime', edgecolors='black', zorder=5)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{obj_id} | {"ACCEPT" if accepted else "REJECT"}')
        ax.view_init(elev=30, azim=135)

        # Equal aspect ratio
        all_pts = pts_base
        mid = all_pts.mean(axis=0)
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)

        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        _dprint(f"  [WARN] save_init_pose_render failed: {e}")

