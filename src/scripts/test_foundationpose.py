from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from src.calibration.io_extrinsics import load_extrinsics_yaml
from src.perception.learned.DINO.dino_identifier import DINOIdentifier, DINOIdentifierConfig
from src.perception.learned.FP.pose_foundation import FoundationPoseConfig, FoundationPoseWrapper
from src.perception.learned.SAM.sam_segmentation import SAMMaskCandidate, SAMSegmenter, SAMSegmenterConfig
from src.perception.ros.multicam_grabber import CameraTopics, MultiCamGrabber


FAST_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
)

CAMERAS = [
    CameraTopics(
        cam_id="zed2i_1",
        depth_topic="/zed2i_1/zed_node/depth/depth_registered",
        info_topic="/zed2i_1/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_1/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_1/zed_node/rgb/color/rect/image/camera_info",
    ),
    CameraTopics(
        cam_id="zed2i_2",
        depth_topic="/zed2i_2/zed_node/depth/depth_registered",
        info_topic="/zed2i_2/zed_node/depth/depth_registered/camera_info",
        rgb_topic="/zed2i_2/zed_node/rgb/color/rect/image",
        rgb_info_topic="/zed2i_2/zed_node/rgb/color/rect/image/camera_info",
    ),
]


@dataclass
class CandidateSelection:
    object_id: str
    score: float
    scores_by_object: dict[str, float]
    candidate: SAMMaskCandidate


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rgb_numpy_to_imgmsg(rgb: np.ndarray, frame_id: str, stamp) -> Image:
    rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    msg = Image()
    msg.header = Header(frame_id=frame_id, stamp=stamp)
    msg.height = int(rgb.shape[0])
    msg.width = int(rgb.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(rgb.shape[1] * 3)
    msg.data = rgb.tobytes()
    return msg


def rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> np.ndarray:
    """
    Returns quaternion as [x, y, z, w].
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    q = np.empty(4, dtype=np.float64)

    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[3] = (R[2, 1] - R[1, 2]) / s
        q[0] = 0.25 * s
        q[1] = (R[0, 1] + R[1, 0]) / s
        q[2] = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[3] = (R[0, 2] - R[2, 0]) / s
        q[0] = (R[0, 1] + R[1, 0]) / s
        q[1] = 0.25 * s
        q[2] = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[3] = (R[1, 0] - R[0, 1]) / s
        q[0] = (R[0, 2] + R[2, 0]) / s
        q[1] = (R[1, 2] + R[2, 1]) / s
        q[2] = 0.25 * s

    q = q / (np.linalg.norm(q) + 1e-12)
    return q.astype(np.float32)


def T_to_pose_stamped(T: np.ndarray, frame_id: str, stamp) -> PoseStamped:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)

    msg = PoseStamped()
    msg.header = Header(frame_id=frame_id, stamp=stamp)

    t = T[:3, 3]
    q = rotation_matrix_to_quaternion_xyzw(T[:3, :3])

    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])

    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.30,
) -> np.ndarray:
    out = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    mask3 = mask.astype(bool)[..., None]
    blended = ((1.0 - alpha) * out + alpha * color_arr).astype(np.uint8)
    return np.where(mask3, blended, out)


def draw_bbox_label(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    text: str,
    color: tuple[int, int, int],
    font_scale: float = 0.6,
) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        out,
        text,
        (x0, max(20, y0 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2,
        cv2.LINE_AA,
    )
    return out


def draw_pose_text(
    image: np.ndarray,
    object_id: str,
    dino_score: float,
    T_object_camera: np.ndarray,
) -> np.ndarray:
    out = image.copy()
    t = T_object_camera[:3, 3]
    lines = [
        f"obj: {object_id}",
        f"dino: {dino_score:.3f}",
        f"tx={t[0]:.4f} ty={t[1]:.4f} tz={t[2]:.4f}",
    ]
    y = 32
    for line in lines:
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        y += 30
    return out


def bbox_crop_with_local_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    crop_rgb = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()
    return crop_rgb, crop_mask


class ProjectedMaskProvider:
    """
    Placeholder for later 3D-cluster projection mode.
    """

    def get_mask(self, view: Any, object_id_hint: str | None = None) -> np.ndarray:
        raise NotImplementedError(
            "Projected mask mode is not wired yet. Connect this to your 3D fused-cluster projection."
        )


class FoundationPoseDebugNode(Node):
    def __init__(self, args: argparse.Namespace, grabber: MultiCamGrabber):
        super().__init__("foundationpose_debug")
        self.args = args
        self.grabber = grabber

        self.output_root = Path(args.output_root).resolve()
        ensure_dir(self.output_root)

        self.palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (255, 128, 0),
            (128, 0, 255),
        ]

        self.last_signature: Optional[tuple[tuple[str, int], ...]] = None
        self.busy = False
        self.frame_counter = 0

        self.mesh_map = {
            "cube": str(Path(args.cad_dir) / "Cube.stl"),
            "Cube": str(Path(args.cad_dir) / "Cube.stl"),
            "red_cube": str(Path(args.cad_dir) / "Cube.stl"),
            "cube_red": str(Path(args.cad_dir) / "Cube.stl"),
            "blue_cube": str(Path(args.cad_dir) / "Cube.stl"),
            "cube_blue": str(Path(args.cad_dir) / "Cube.stl"),
        }

        self.dino = DINOIdentifier(
            DINOIdentifierConfig(
                model_name=args.dino_model_name,
                device=args.device,
                reference_dir=args.reference_dir,
                use_masked_background=True,
            )
        )
        self.get_logger().info("Building DINO reference bank...")
        self.dino.build_reference_bank_from_folder()
        self.get_logger().info(f"DINO reference bank size: {len(self.dino.reference_bank)}")

        self.sam: SAMSegmenter | None = None
        if args.mask_source == "sam":
            self.sam = SAMSegmenter(
                SAMSegmenterConfig(
                    repo_root=args.sam_repo_root,
                    checkpoint=args.sam_checkpoint,
                    model_cfg=args.sam_model_cfg,
                    device=args.device,
                    max_image_side=args.sam_max_image_side,
                    min_mask_area=args.sam_min_mask_area,
                    min_bbox_side_px=args.sam_min_bbox_side_px,
                    attach_rgb_crops=False,
                )
            )

        self.projected_provider = ProjectedMaskProvider()

        self.fp = FoundationPoseWrapper(
            FoundationPoseConfig(
                repo_root=args.fp_repo_root,
                weights_dir=args.fp_weights_dir,
                debug_dir=str(self.output_root / "fp_internal_debug"),
                debug=args.fp_debug,
                est_refine_iter=args.est_refine_iter,
            )
        )

        self.pub_raw = {}
        self.pub_sam_overlay = {}
        self.pub_candidate_overview = {}
        self.pub_dino_overlay = {}
        self.pub_pose_overlay = {}
        self.pub_pose = {}

        for c in CAMERAS:
            cam_id = c.cam_id
            self.pub_raw[cam_id] = self.create_publisher(
                Image, f"/perception/debug/fp/rgb_raw/{cam_id}", FAST_QOS
            )
            self.pub_sam_overlay[cam_id] = self.create_publisher(
                Image, f"/perception/debug/fp/sam_overlay/{cam_id}", FAST_QOS
            )
            self.pub_candidate_overview[cam_id] = self.create_publisher(
                Image, f"/perception/debug/fp/candidate_overview/{cam_id}", FAST_QOS
            )
            self.pub_dino_overlay[cam_id] = self.create_publisher(
                Image, f"/perception/debug/fp/dino_overlay/{cam_id}", FAST_QOS
            )
            self.pub_pose_overlay[cam_id] = self.create_publisher(
                Image, f"/perception/debug/fp/pose_overlay/{cam_id}", FAST_QOS
            )
            self.pub_pose[cam_id] = self.create_publisher(
                PoseStamped, f"/perception/debug/fp/object_pose/{cam_id}", FAST_QOS
            )

        self.timer = self.create_timer(args.timer_period_s, self._tick)
        self.get_logger().info("FoundationPoseDebugNode started")

    def _views_signature(self, views: list[Any]) -> tuple[tuple[str, int], ...]:
        items = []
        for v in views:
            stamp_ns = int(float(v.stamp_s) * 1e9)
            items.append((str(v.cam_id), stamp_ns))
        items.sort(key=lambda x: x[0])
        return tuple(items)

    def _resolve_mesh_path(self, object_id: str) -> str:
        if object_id in self.mesh_map:
            return self.mesh_map[object_id]

        direct = Path(self.args.cad_dir) / f"{object_id}.stl"
        if direct.exists():
            return str(direct)

        raise FileNotFoundError(f"No CAD model mapping found for object_id='{object_id}'")

    def _draw_candidate_overview(
        self,
        rgb: np.ndarray,
        masks: list[SAMMaskCandidate],
        max_masks: int,
    ) -> np.ndarray:
        vis = rgb.copy()
        for i, cand in enumerate(masks[:max_masks]):
            color = self.palette[i % len(self.palette)]
            vis = draw_mask_overlay(vis, cand.mask, color=color, alpha=0.22)
            txt = f"{i}: sam={cand.score:.2f} area={cand.area}"
            vis = draw_bbox_label(vis, cand.bbox_xyxy, txt, color, font_scale=0.55)
        return vis

    def _classify_single_candidate(
        self,
        rgb: np.ndarray,
        cand: SAMMaskCandidate,
    ) -> CandidateSelection | None:
        crop_rgb, crop_mask = bbox_crop_with_local_mask(rgb, cand.mask, cand.bbox_xyxy)
        if crop_rgb.size == 0 or int(crop_mask.sum()) == 0:
            return None

        try:
            res = self.dino.classify_crop(crop_rgb, mask=crop_mask)
        except Exception as e:
            self.get_logger().warn(f"DINO classification failed on SAM candidate: {e}")
            return None

        best_score = float(res.score)
        sorted_scores = sorted(res.scores_by_object.items(), key=lambda kv: kv[1], reverse=True)
        second_score = float(sorted_scores[1][1]) if len(sorted_scores) > 1 else -1.0
        margin = best_score - second_score

        object_id = res.object_id
        if best_score < self.args.dino_min_score:
            object_id = "unknown"
        if self.args.dino_min_margin > 0.0 and margin < self.args.dino_min_margin:
            object_id = "unknown"

        return CandidateSelection(
            object_id=object_id,
            score=best_score,
            scores_by_object={k: float(v) for k, v in res.scores_by_object.items()},
            candidate=cand,
        )

    def _classify_sam_candidates(
        self,
        rgb: np.ndarray,
        masks: list[SAMMaskCandidate],
    ) -> list[CandidateSelection]:
        out: list[CandidateSelection] = []
        for cand in masks:
            sel = self._classify_single_candidate(rgb, cand)
            if sel is not None:
                out.append(sel)
        out.sort(key=lambda x: x.score, reverse=True)
        return out

    def _draw_dino_overlay(
        self,
        rgb: np.ndarray,
        ranked: list[CandidateSelection],
        max_candidates: int,
    ) -> np.ndarray:
        vis = rgb.copy()
        for i, sel in enumerate(ranked[:max_candidates]):
            color = self.palette[i % len(self.palette)]
            vis = draw_mask_overlay(vis, sel.candidate.mask, color=color, alpha=0.22)
            txt = f"{sel.object_id} {sel.score:.2f}"
            vis = draw_bbox_label(vis, sel.candidate.bbox_xyxy, txt, color, font_scale=0.6)
        return vis

    def _select_mask_and_build_overlays(
        self,
        view: Any,
    ) -> tuple[CandidateSelection, np.ndarray, np.ndarray, np.ndarray]:
        rgb = view.rgb

        if self.args.mask_source == "sam":
            assert self.sam is not None
            masks = self.sam.generate_auto(rgb)
            if not masks:
                raise RuntimeError(f"No SAM masks found for {view.cam_id}")

            sam_overlay = self._draw_candidate_overview(rgb, masks, self.args.max_candidate_draw)
            ranked = self._classify_sam_candidates(rgb, masks)
            if not ranked:
                raise RuntimeError(f"No SAM candidates could be classified by DINO for {view.cam_id}")

            candidate_overview = self._draw_candidate_overview(rgb, masks, self.args.max_candidate_draw)
            dino_overlay = self._draw_dino_overlay(rgb, ranked, self.args.max_candidate_draw)

            filtered = ranked
            if self.args.target_object:
                filtered = [r for r in filtered if r.object_id == self.args.target_object]

            filtered = [r for r in filtered if r.object_id != "unknown"]

            if not filtered:
                raise RuntimeError(
                    f"No valid SAM+DINO candidate left for {view.cam_id} "
                    f"(target_object={self.args.target_object})"
                )

            return filtered[0], sam_overlay, candidate_overview, dino_overlay

        if self.args.mask_source == "projected":
            if not self.args.target_object:
                raise RuntimeError("Projected mode currently requires --target-object")
            mask = self.projected_provider.get_mask(view=view, object_id_hint=self.args.target_object)
            fake = SAMMaskCandidate(
                mask=mask.astype(bool),
                score=1.0,
                bbox_xyxy=(0, 0, mask.shape[1], mask.shape[0]),
                area=int(mask.sum()),
                crop_rgb=None,
            )
            sel = CandidateSelection(
                object_id=self.args.target_object,
                score=1.0,
                scores_by_object={self.args.target_object: 1.0},
                candidate=fake,
            )

            sam_overlay = draw_mask_overlay(rgb, fake.mask, color=(0, 255, 0), alpha=0.25)
            candidate_overview = sam_overlay.copy()
            dino_overlay = sam_overlay.copy()
            return sel, sam_overlay, candidate_overview, dino_overlay

        raise ValueError(f"Unsupported mask_source: {self.args.mask_source}")

    def _save_outputs(
        self,
        *,
        view: Any,
        selection: CandidateSelection,
        sam_overlay: np.ndarray,
        candidate_overview: np.ndarray,
        dino_overlay: np.ndarray,
        pose_overlay: np.ndarray,
        pose_T: np.ndarray,
    ) -> None:
        cam_dir = self.output_root / view.cam_id
        ensure_dir(cam_dir)

        stamp_ns = int(float(view.stamp_s) * 1e9)
        stem = str(stamp_ns)

        cv2.imwrite(str(cam_dir / f"{stem}_rgb.png"), cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(cam_dir / f"{stem}_sam_overlay.png"), cv2.cvtColor(sam_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(
            str(cam_dir / f"{stem}_candidate_overview.png"),
            cv2.cvtColor(candidate_overview, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(cam_dir / f"{stem}_dino_overlay.png"), cv2.cvtColor(dino_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(cam_dir / f"{stem}_pose_overlay.png"), cv2.cvtColor(pose_overlay, cv2.COLOR_RGB2BGR))
        np.save(cam_dir / f"{stem}_T_object_camera.npy", pose_T)

        payload = {
            "cam_id": view.cam_id,
            "stamp_s": float(view.stamp_s),
            "object_id": selection.object_id,
            "dino_score": float(selection.score),
            "scores_by_object": selection.scores_by_object,
            "mask_area": int(selection.candidate.mask.sum()),
            "bbox_xyxy": [int(v) for v in selection.candidate.bbox_xyxy],
            "T_object_camera": pose_T.tolist(),
        }
        (cam_dir / f"{stem}_result.json").write_text(json.dumps(payload, indent=2))

    def _publish_outputs(
        self,
        *,
        view: Any,
        selection: CandidateSelection,
        sam_overlay: np.ndarray,
        candidate_overview: np.ndarray,
        dino_overlay: np.ndarray,
        pose_overlay: np.ndarray,
        pose_T: np.ndarray,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        cam_id = view.cam_id

        self.pub_raw[cam_id].publish(
            rgb_numpy_to_imgmsg(view.rgb, frame_id=cam_id, stamp=stamp)
        )
        self.pub_sam_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(sam_overlay, frame_id=cam_id, stamp=stamp)
        )
        self.pub_candidate_overview[cam_id].publish(
            rgb_numpy_to_imgmsg(candidate_overview, frame_id=cam_id, stamp=stamp)
        )
        self.pub_dino_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(dino_overlay, frame_id=cam_id, stamp=stamp)
        )
        self.pub_pose_overlay[cam_id].publish(
            rgb_numpy_to_imgmsg(pose_overlay, frame_id=cam_id, stamp=stamp)
        )
        self.pub_pose[cam_id].publish(
            T_to_pose_stamped(pose_T, frame_id=cam_id, stamp=stamp)
        )

    def _process_single_view(self, view: Any) -> None:
        if view.rgb is None:
            self.get_logger().warn(f"{view.cam_id}: missing RGB, skipping")
            return
        if view.depth is None:
            self.get_logger().warn(f"{view.cam_id}: missing depth, skipping")
            return

        if view.rgb.shape[:2] != view.depth.shape[:2]:
            self.get_logger().warn(
                f"{view.cam_id}: rgb/depth shape mismatch {view.rgb.shape[:2]} vs {view.depth.shape[:2]}, skipping"
            )
            return

        rgb = view.rgb
        depth = view.depth
        K = np.asarray(view.K, dtype=np.float32).reshape(3, 3)

        t0 = time.perf_counter()
        selection, sam_overlay, candidate_overview, dino_overlay = self._select_mask_and_build_overlays(view)
        t1 = time.perf_counter()

        mesh_path = self._resolve_mesh_path(selection.object_id)

        t2 = time.perf_counter()
        result = self.fp.estimate_pose(
            object_id=selection.object_id,
            mesh_path=mesh_path,
            rgb=rgb,
            depth=depth,
            K=K,
            mask=selection.candidate.mask,
        )
        t3 = time.perf_counter()

        pose_overlay = draw_mask_overlay(rgb, selection.candidate.mask, color=(0, 255, 0), alpha=0.28)
        pose_overlay = draw_bbox_label(
            pose_overlay,
            selection.candidate.bbox_xyxy,
            f"{selection.object_id} {selection.score:.2f}",
            (0, 255, 0),
            font_scale=0.7,
        )
        pose_overlay = draw_pose_text(
            pose_overlay,
            selection.object_id,
            selection.score,
            result.T_object_camera,
        )

        self._save_outputs(
            view=view,
            selection=selection,
            sam_overlay=sam_overlay,
            candidate_overview=candidate_overview,
            dino_overlay=dino_overlay,
            pose_overlay=pose_overlay,
            pose_T=result.T_object_camera,
        )

        self._publish_outputs(
            view=view,
            selection=selection,
            sam_overlay=sam_overlay,
            candidate_overview=candidate_overview,
            dino_overlay=dino_overlay,
            pose_overlay=pose_overlay,
            pose_T=result.T_object_camera,
        )

        self.get_logger().info(
            f"[{view.cam_id}] obj={selection.object_id} "
            f"dino={selection.score:.3f} "
            f"mask_area={int(selection.candidate.mask.sum())} "
            f"select={(t1 - t0) * 1000:.1f} ms "
            f"fp={(t3 - t2) * 1000:.1f} ms"
        )
        self.get_logger().info(f"[{view.cam_id}] T_object_camera=\n{result.T_object_camera}")

    def _tick(self) -> None:
        if self.busy:
            return

        views = self.grabber.get_latest_views()
        if views is None:
            return

        signature = self._views_signature(views)
        if signature == self.last_signature:
            return
        self.last_signature = signature

        self.busy = True
        try:
            self.frame_counter += 1
            for view in views:
                try:
                    self._process_single_view(view)
                except Exception as e:
                    self.get_logger().warn(f"[{view.cam_id}] processing failed: {e}")
        finally:
            self.busy = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--device", default="cuda")
    p.add_argument("--mask-source", choices=["sam", "projected"], default="sam")
    p.add_argument("--target-object", default=None)

    p.add_argument("--reference-dir", default="Data/ZED_screens")
    p.add_argument("--cad-dir", default="Data/CAD_Models")
    p.add_argument("--output-root", default="outputs/foundationpose")

    p.add_argument("--dino-model-name", default="dinov2_vitb14")
    p.add_argument("--dino-min-score", type=float, default=0.70)
    p.add_argument("--dino-min-margin", type=float, default=0.00)

    p.add_argument("--sam-repo-root", default="external/sam2")
    p.add_argument("--sam-checkpoint", default="external/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    p.add_argument("--sam-model-cfg", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    p.add_argument("--sam-max-image-side", type=int, default=1024)
    p.add_argument("--sam-min-mask-area", type=int, default=1500)
    p.add_argument("--sam-min-bbox-side-px", type=int, default=20)

    p.add_argument("--fp-repo-root", default="external/FoundationPose")
    p.add_argument("--fp-weights-dir", default="external/FoundationPose/weights")
    p.add_argument("--fp-debug", type=int, default=2)
    p.add_argument("--est-refine-iter", type=int, default=5)

    p.add_argument("--timer-period-s", type=float, default=0.25)
    p.add_argument("--max-candidate-draw", type=int, default=8)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    rclpy.init()

    T_map = load_extrinsics_yaml("config/camera_extrinsics.yaml")
    grabber = MultiCamGrabber(
        cameras=CAMERAS,
        sync_slop_s=0.10,
        use_best_effort_if_unsynced=True,
        static_extrinsics_base_cam=T_map,
        rgb_depth_max_dt_s=0.08,
    )

    node = FoundationPoseDebugNode(args=args, grabber=grabber)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(grabber)
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        executor.remove_node(node)
        executor.remove_node(grabber)
        node.destroy_node()
        grabber.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()