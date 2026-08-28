import json
import os

import numpy as np
import trimesh
from matplotlib.path import Path
from scipy.spatial import ConvexHull, QhullError

# ---- CONFIGURATION ----

ASSET_DIR = "./Data/CAD_Models_centered/plumbers_block"
OUTPUT_FILE = "pb_stable_poses.json"

VALID_EXTENSIONS = (".obj", ".stl", ".ply")

# Geometric tolerances
PLANE_TOLERANCE_FRACTION = 0.01
DIST_TOLERANCE_FRACTION = 0.005

# Candidate face grouping
ANGLE_TOLERANCE = np.radians(3.0)

# Perturbation used to determine rotational stability
PERTURBATION_ANGLE = np.radians(1.0)

# Numerical tolerance for classifying COM height
HEIGHT_TOLERANCE_FRACTION = 1e-4

# Minimum number of contact vertices required
MIN_CONTACT_POINTS = 3

# Free-rotation-axis search: always probe the vertical (yaw) axis,
# plus a sweep of horizontal axes so that non-axis-aligned free
# rotations (e.g. a part rolling on a cylindrical shaft) are found
# too, not just world-X/Y tilts.
HORIZONTAL_AXIS_STEP_DEG = 5.0

# Neutral hits from adjacent swept azimuths that belong to the same
# physical axis get merged into a single reported axis.
AXIS_MERGE_TOLERANCE_DEG = 7.5


# ---- BASIC TRANSFORMS ----


def align_vector_to_target(vec, target=None):
    """
    Return a rotation matrix that rotates vec onto target.
    """

    if target is None:
        target = np.array([0.0, 0.0, -1.0])

    # IMPORTANT:
    # Use copy=True because trimesh may return read-only arrays.
    vec = np.array(vec, dtype=float, copy=True)
    target = np.array(target, dtype=float, copy=True)

    vec_norm = np.linalg.norm(vec)
    target_norm = np.linalg.norm(target)

    if vec_norm < 1e-12:
        raise ValueError("Cannot normalize zero-length vector.")

    if target_norm < 1e-12:
        raise ValueError("Cannot normalize zero-length target.")

    vec /= vec_norm
    target /= target_norm

    v = np.cross(vec, target)
    c = np.dot(vec, target)

    if np.isclose(c, 1.0):
        return np.eye(4)

    if np.isclose(c, -1.0):
        # 180 degree rotation around any axis perpendicular
        # to vec.
        axis = np.array([1.0, 0.0, 0.0], dtype=float)

        if abs(np.dot(axis, vec)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)

        axis -= np.dot(axis, vec) * vec

        axis /= np.linalg.norm(axis)

        return trimesh.transformations.rotation_matrix(np.pi, axis)

    s = np.linalg.norm(v)

    kmat = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    r = np.eye(3) + kmat + kmat @ kmat * ((1.0 - c) / (s**2))

    transform = np.eye(4)
    transform[:3, :3] = r

    return transform


# ---- PLANE DETECTION ----


def get_unique_planes(hull, dist_tolerance):
    """
    Group convex-hull faces that belong to the same geometric plane.

    Returns:
        [(normal, offset), ...]
    """

    normals = hull.face_normals

    face_points = hull.vertices[hull.faces[:, 0]]

    offsets = np.einsum("ij,ij->i", normals, face_points)

    unique_planes = []

    for normal, offset in zip(normals, offsets):
        matched = False

        for unique_normal, unique_offset in unique_planes:
            dot_val = np.clip(np.dot(normal, unique_normal), -1.0, 1.0)

            angle = np.arccos(dot_val)

            if angle < ANGLE_TOLERANCE and abs(offset - unique_offset) < dist_tolerance:
                matched = True
                break

        if not matched:
            unique_planes.append((normal, offset))

    return unique_planes


# ---- CONTACT EXTRACTION ----


def get_contact_polygon(mesh, plane_tolerance):
    """
    Find the part of the mesh touching the table z=0.

    Returns:
        poly_2d, contact_area

    or

        None, 0
    """

    z = mesh.vertices[:, 2]

    contact_mask = np.abs(z) <= plane_tolerance

    contact_points = mesh.vertices[contact_mask]

    if len(contact_points) < MIN_CONTACT_POINTS:
        return None, 0.0

    contact_xy = contact_points[:, :2]

    # Remove duplicate points
    contact_xy = np.unique(np.round(contact_xy, decimals=10), axis=0)

    if len(contact_xy) < MIN_CONTACT_POINTS:
        return None, 0.0

    try:
        hull = ConvexHull(contact_xy)

        polygon = contact_xy[hull.vertices]

        area = float(hull.volume)

        return polygon, area

    except QhullError:
        # Degenerate/collinear contact points (e.g. a near-point or
        # near-line contact) can't form a 2D hull.
        return None, 0.0


# ---- SUPPORT / STATIC STABILITY ----


def point_inside_polygon(point, polygon, tolerance):
    """
    Test whether point lies inside or sufficiently close to polygon.
    """

    if polygon is None or len(polygon) < 3:
        return False

    path = Path(polygon)

    return path.contains_point(point[:2], radius=tolerance)


def compute_stability_margin(com, polygon):
    """
    Minimum distance from projected COM to support-polygon edges.
    """

    if polygon is None or len(polygon) < 3:
        return -np.inf

    distances = []

    p = com[:2]

    for i in range(len(polygon)):
        a = polygon[i - 1]
        b = polygon[i]

        edge = b - a

        denominator = np.linalg.norm(edge)

        if denominator < 1e-12:
            continue

        distance = abs(np.cross(edge, p - a)) / denominator

        distances.append(distance)

    if not distances:
        return -np.inf

    return float(min(distances))


# ---- POSE EVALUATION ----


def evaluate_pose(mesh, transform, plane_tolerance):
    """
    Evaluate whether a transformed mesh is statically stable.

    Returns a dictionary containing:
        stable
        contact polygon
        contact area
        COM
        COM height
        stability margin
    """

    posed_mesh = mesh.copy().apply_transform(transform)

    com = trimesh.transformations.transform_points([mesh.center_mass], transform)[0]

    polygon, contact_area = get_contact_polygon(posed_mesh, plane_tolerance)

    if polygon is None:
        return {
            "stable": False,
            "contact_polygon": None,
            "contact_area": 0.0,
            "com": com,
            "com_height": float(com[2]),
            "margin": -np.inf,
        }

    stable = point_inside_polygon(com, polygon, plane_tolerance)

    margin = compute_stability_margin(com, polygon)

    return {
        "stable": stable,
        "contact_polygon": polygon,
        "contact_area": contact_area,
        "com": com,
        "com_height": float(com[2]),
        "margin": margin,
    }


# ---- ROTATIONAL STABILITY ANALYSIS ----


def perturb_pose_about_object_axis(pose_transform, axis_world, angle, center_world):
    """
    Rotate the entire object around a world-space axis
    passing through center_world.
    """

    axis_world = np.asarray(axis_world, dtype=float)

    axis_world /= np.linalg.norm(axis_world)

    T_to_origin = trimesh.transformations.translation_matrix(-center_world)

    R = trimesh.transformations.rotation_matrix(angle, axis_world)

    T_back = trimesh.transformations.translation_matrix(center_world)

    return T_back @ R @ T_to_origin @ pose_transform


def rebase_to_table(mesh, transform):
    """
    After perturbing a pose, translate the object so that its
    lowest point touches z=0 again.
    """

    test_mesh = mesh.copy().apply_transform(transform)

    z_min = np.min(test_mesh.vertices[:, 2])

    T = trimesh.transformations.translation_matrix([0.0, 0.0, -z_min])

    return T @ transform


def classify_rotation_stability(
    mesh, pose_transform, axis, plane_tolerance, height_tolerance
):
    """
    Determine whether rotation about a given axis is:

        restoring
        neutral
        unstable
        constrained

    The pose is perturbed by +/- epsilon.

    Important:
        We do NOT use symmetry or inertia here.

    The test is based on the physical equilibrium criterion:

        COM projection must remain within the support region.

    Additionally, COM height is examined to distinguish
    restoring from unstable behavior.
    """

    initial = evaluate_pose(mesh, pose_transform, plane_tolerance)

    if not initial["stable"]:
        return {"status": "invalid_initial_pose", "plus": None, "minus": None}

    com_world = initial["com"]

    axis = np.array(axis, dtype=float, copy=True)

    results = []

    for sign in (+1.0, -1.0):
        perturbed = perturb_pose_about_object_axis(
            pose_transform, axis, sign * PERTURBATION_ANGLE, com_world
        )

        # Put the lowest point back on the table.
        perturbed = rebase_to_table(mesh, perturbed)

        result = evaluate_pose(mesh, perturbed, plane_tolerance)

        results.append(result)

    plus_result = results[0]
    minus_result = results[1]

    plus_dh = plus_result["com_height"] - initial["com_height"]

    minus_dh = minus_result["com_height"] - initial["com_height"]

    plus_stable = plus_result["stable"]
    minus_stable = minus_result["stable"]

    # --------------------------------------------------------
    # Both directions remain stable and COM height does not
    # change appreciably -> neutral DOF.
    # --------------------------------------------------------

    if (
        plus_stable
        and minus_stable
        and abs(plus_dh) <= height_tolerance
        and abs(minus_dh) <= height_tolerance
    ):
        status = "neutral"

    # --------------------------------------------------------
    # Both perturbations increase COM height.
    #
    # Gravity therefore produces a restoring tendency.
    # --------------------------------------------------------

    elif plus_dh > height_tolerance and minus_dh > height_tolerance:
        status = "restoring"

    # --------------------------------------------------------
    # At least one perturbation decreases COM height.
    # That direction is energetically downhill.
    # --------------------------------------------------------

    elif plus_dh < -height_tolerance or minus_dh < -height_tolerance:
        status = "unstable"

    # --------------------------------------------------------
    # Otherwise the classification is ambiguous, usually
    # because of mesh resolution/contact tolerances.
    # --------------------------------------------------------

    else:
        status = "constrained"

    return {
        "status": status,
        "plus": {
            "stable": bool(plus_stable),
            "com_height_change": float(plus_dh),
            "stability_margin": float(plus_result["margin"]),
        },
        "minus": {
            "stable": bool(minus_stable),
            "com_height_change": float(minus_dh),
            "stability_margin": float(minus_result["margin"]),
        },
    }


# ---- ORIENTATION DOF ANALYSIS ----


def build_candidate_rotation_axes(step_deg=HORIZONTAL_AXIS_STEP_DEG):
    """
    Candidate rotation axes to probe for a given stable pose:
    the vertical (yaw) axis, plus a sweep of horizontal axes.

    A horizontal axis and its opposite describe the same physical
    line, so azimuths only need to span [0, 180) degrees. This lets
    a free "rolling" axis be found even when it isn't aligned with
    world X or Y (e.g. a part resting on a cylindrical section).
    """

    candidates = [("Rz", np.array([0.0, 0.0, 1.0]))]

    for az_deg in np.arange(0.0, 180.0, step_deg):
        rad = np.radians(az_deg)

        axis = np.array([np.cos(rad), np.sin(rad), 0.0])

        candidates.append((f"R_horiz_{az_deg:.0f}deg", axis))

    return candidates


def merge_neutral_axes(hits, merge_tolerance_deg):
    """
    Merge neutral-axis hits that describe the same physical axis.

    Dense azimuth sampling can trigger "neutral" on several
    adjacent samples around one real rolling axis; these are
    collapsed into a single representative entry. Axis and
    opposite-axis hits (antiparallel) are treated as the same line.
    """

    merged = []

    for name, axis in hits:
        matched = False

        for group in merged:
            dot = np.clip(abs(np.dot(axis, group["_axis"])), -1.0, 1.0)

            angle_deg = np.degrees(np.arccos(dot))

            if angle_deg < merge_tolerance_deg:
                matched = True
                break

        if not matched:
            merged.append({"name": name, "axis_world": axis.tolist(), "_axis": axis})

    for group in merged:
        del group["_axis"]

    return merged


def analyze_orientation_dofs(mesh, pose_transform, plane_tolerance, height_tolerance):
    """
    Determine which rotation axes are physically free (neutral)
    for a stable pose: the vertical yaw axis, plus any horizontal
    axis the part can roll about.

    Returns explicit information instead of attempting to
    infer DOFs from symmetry. Only neutral hits are kept in
    detail; restoring/unstable candidates are not physically
    interesting here and are discarded to keep the output compact.
    """

    initial = evaluate_pose(mesh, pose_transform, plane_tolerance)

    if not initial["stable"]:
        return {
            "stable_orientation_dofs": [],
            "neutral_orientation_axes": [],
            "orientation_analysis": {},
        }

    analysis = {}
    neutral_hits = []

    for name, axis in build_candidate_rotation_axes():
        result = classify_rotation_stability(
            mesh, pose_transform, axis, plane_tolerance, height_tolerance
        )

        if result["status"] == "neutral":
            neutral_hits.append((name, axis))
            analysis[name] = result

    neutral_axes = merge_neutral_axes(neutral_hits, AXIS_MERGE_TOLERANCE_DEG)

    return {
        "stable_orientation_dofs": [a["name"] for a in neutral_axes],
        "neutral_orientation_axes": neutral_axes,
        "orientation_analysis": analysis,
    }


# ---- TRANSLATIONAL DOF DESCRIPTION ----


def determine_contact_type(contact_area, polygon):
    """
    Simple contact classification.
    """

    if polygon is None or len(polygon) == 0:
        return "none"

    if len(polygon) == 1:
        return "point"

    if len(polygon) == 2:
        return "line"

    if contact_area < 1e-8:
        return "degenerate_area"

    return "surface"


# ---- PROCESS ONE MESH ----


def process_mesh(file_path):

    mesh = trimesh.load(file_path)

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump()[0]

    mesh.remove_unreferenced_vertices()

    bbox_extent = float(np.max(mesh.extents))

    plane_tolerance = max(1e-3, bbox_extent * PLANE_TOLERANCE_FRACTION)

    dist_tolerance = max(1e-3, bbox_extent * DIST_TOLERANCE_FRACTION)

    height_tolerance = max(1e-6, bbox_extent * HEIGHT_TOLERANCE_FRACTION)

    hull = mesh.convex_hull

    unique_planes = get_unique_planes(hull, dist_tolerance)

    print(f"  └─ Extent: {bbox_extent:.2f} | Candidate planes: {len(unique_planes)}")

    stable_poses = []

    rejected_contacts = 0
    rejected_com = 0

    for idx, (normal, _) in enumerate(unique_planes):
        # ----------------------------------------------------
        # Orient candidate supporting plane downward.
        # ----------------------------------------------------

        T_rot = align_vector_to_target(normal, np.array([0.0, 0.0, -1.0]))

        temp_mesh = mesh.copy().apply_transform(T_rot)

        # Put lowest point on table.
        z_min = np.min(temp_mesh.vertices[:, 2])

        T_trans = trimesh.transformations.translation_matrix([0.0, 0.0, -z_min])

        T_full = T_trans @ T_rot

        # ----------------------------------------------------
        # Evaluate candidate.
        # ----------------------------------------------------

        result = evaluate_pose(mesh, T_full, plane_tolerance)

        if result["contact_polygon"] is None:
            rejected_contacts += 1
            continue

        if not result["stable"]:
            rejected_com += 1
            continue

        poly_2d = result["contact_polygon"]

        contact_area = result["contact_area"]

        margin = result["margin"]

        contact_type = determine_contact_type(contact_area, poly_2d)

        # ----------------------------------------------------
        # Determine physical orientation DOFs.
        # ----------------------------------------------------

        dof_info = analyze_orientation_dofs(
            mesh, T_full, plane_tolerance, height_tolerance
        )

        # ----------------------------------------------------
        # Translation on a horizontal table.
        #
        # We report Tx/Ty separately from orientation DOFs.
        #
        # They describe placement on the table, NOT
        # rotational stability.
        # ----------------------------------------------------

        table_translation_dofs = []

        if contact_type == "surface":
            table_translation_dofs = ["Tx", "Ty"]

        # ----------------------------------------------------
        # Store result.
        # ----------------------------------------------------

        stable_poses.append(
            {
                "pose_id": idx,
                "transform": T_full.tolist(),
                "contact_polygon_2d": poly_2d.tolist(),
                "contact_area": float(contact_area),
                "contact_type": contact_type,
                "stability_margin": float(margin),
                "stable_orientation_dofs": dof_info["stable_orientation_dofs"],
                "neutral_orientation_axes": dof_info["neutral_orientation_axes"],
                "table_translation_dofs": table_translation_dofs,
                "orientation_analysis": dof_info["orientation_analysis"],
            }
        )

    # --------------------------------------------------------
    # Remove duplicate stable poses.
    #
    # Several coplanar hull triangles can produce the same
    # orientation.
    # --------------------------------------------------------

    unique_results = []

    for pose in stable_poses:
        R = np.array(pose["transform"])[:3, :3]

        is_duplicate = False

        for other in unique_results:
            R_other = np.array(other["transform"])[:3, :3]

            R_diff = R_other.T @ R

            trace = np.trace(R_diff)

            angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

            if angle < np.radians(1.0):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_results.append(pose)

    print(
        f"  └─ Results: "
        f"{len(unique_results)} stable poses "
        f"({rejected_contacts} bad contacts, "
        f"{rejected_com} unstable CoMs)"
    )

    return unique_results


# ---- MAIN ----


def main():

    if not os.path.exists(ASSET_DIR):
        print(f"Error: Directory '{ASSET_DIR}' does not exist.")

        return

    mesh_files = [
        f for f in os.listdir(ASSET_DIR) if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not mesh_files:
        print("No valid mesh files found.")

        return

    print(f"Found {len(mesh_files)} mesh files in {ASSET_DIR}\n")

    results = {}

    for filename in mesh_files:
        path = os.path.join(ASSET_DIR, filename)

        print(f"Processing: {filename}")

        try:
            results[filename] = process_mesh(path)

        except Exception as exc:
            print(f"  ERROR: {exc}")

            results[filename] = []

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=4)

    print("\nAnalysis complete.")

    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
