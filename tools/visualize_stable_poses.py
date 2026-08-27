import json
import os

import numpy as np
import trimesh

# ---- CONFIGURATION ----

ASSET_DIR = "./Data/CAD_Models_centered/plumbers_block"
INPUT_FILE = "pb_stable_poses.json"

MAX_SAMPLES = 5

# ------------------------------------------------------------
# Visualization parameters
# ------------------------------------------------------------

TABLE_GRID_SIZE_FACTOR = 3.0
TABLE_GRID_COUNT = 10

CONTACT_ALPHA = 160
OBJECT_ALPHA = 255

DOF_COLOR = [0, 255, 80, 255]
CONTACT_COLOR = [255, 0, 0, CONTACT_ALPHA]
COM_COLOR = [255, 255, 0, 255]

GHOST_COLOR = [100, 180, 255, 80]

# Ghost sampling step for visualizing the manifold swept out by a
# free rotational DOF.
GHOST_ANGLE_STEP_DEG = 20


# ---- UTILITY ----


def normalize(v):
    """Return normalized vector, or None if vector is degenerate."""
    v = np.asarray(v, dtype=float)

    n = np.linalg.norm(v)

    if n < 1e-10:
        return None

    return v / n


# ---- PHYSICAL DOF ANALYSIS ----


def determine_physical_dofs(pose):
    """
    Build the physical free-DOF summary directly from the fields
    compute_stable_poses.py actually writes: table_translation_dofs
    for Tx/Ty, and neutral_orientation_axes for any rotation axis
    (vertical yaw or a swept-in horizontal rolling axis) found to
    be physically neutral for this pose.
    """

    free_dofs = []

    for name in pose.get("table_translation_dofs", []):
        axis = np.array([1.0, 0.0, 0.0]) if name == "Tx" else np.array([0.0, 1.0, 0.0])

        free_dofs.append({"type": "translation", "name": name, "axis": axis})

    for entry in pose.get("neutral_orientation_axes", []):
        free_dofs.append(
            {
                "type": "rotation",
                "name": entry["name"],
                "axis": np.array(entry["axis_world"], dtype=float),
            }
        )

    return {"dof_count": len(free_dofs), "free_dofs": free_dofs}


# ---- ARROW CREATION ----

def create_dof_text_labels(center, dof_info, axis_len):
    """
    Generates 3D text geometry displaying numerical vectors 
    directly in the visualizer window.
    """
    labels = []
    rot_dofs = [d for d in dof_info["free_dofs"] if d["type"] == "rotation"]

    for idx, dof in enumerate(rot_dofs):
        axis_vec = np.round(dof["axis"], 3).tolist()
        label_text = f"{dof['name']}: {axis_vec}"

        try:
            # Render text string as a 3D surface mesh
            text_mesh = trimesh.creation.text_mesh(label_text, height=axis_len * 0.08)
            text_mesh.visual.face_colors = [0, 255, 80, 255]  # Green color match

            # Position label stacked above the Center of Mass
            offset = center + np.array([0.0, 0.0, axis_len * (0.6 + idx * 0.15)])
            text_mesh.apply_translation(offset)

            labels.append(text_mesh)
        except Exception:
            # Fallback if optional 3D text rendering dependencies are missing
            pass

    return labels


def create_3d_arrow(start, direction, length, radius, color=DOF_COLOR):
    start = np.asarray(start, dtype=float)
    direction = normalize(direction)

    if direction is None:
        return None

    shaft_len = length * 0.75
    head_len = length * 0.25
    head_radius = radius * 2.2

    shaft = trimesh.creation.cylinder(radius=radius, height=shaft_len, sections=16)

    shaft.apply_translation([0.0, 0.0, shaft_len / 2.0])

    head = trimesh.creation.cone(radius=head_radius, height=head_len, sections=16)

    head.apply_translation([0.0, 0.0, shaft_len + head_len / 2.0])

    arrow = trimesh.util.concatenate([shaft, head])

    arrow.visual.face_colors = color

    z_axis = np.array([0.0, 0.0, 1.0])

    if np.allclose(direction, z_axis):
        rotation = np.eye(4)

    elif np.allclose(direction, -z_axis):
        rotation = trimesh.transformations.rotation_matrix(np.pi, [1.0, 0.0, 0.0])

    else:
        rotation_axis = np.cross(z_axis, direction)

        rotation_axis = normalize(rotation_axis)

        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))

        rotation = trimesh.transformations.rotation_matrix(angle, rotation_axis)

    translation = trimesh.transformations.translation_matrix(start)

    arrow.apply_transform(translation @ rotation)

    return arrow


# ---- TRANSLATIONAL DOF VISUALIZATION ----


def create_translation_dof_arrows(center, dof_info, axis_length):
    arrows = []

    radius = axis_length * 0.035
    arrow_length = axis_length * 0.7

    for dof in dof_info["free_dofs"]:
        if dof["type"] != "translation":
            continue

        axis = normalize(dof["axis"])

        if axis is None:
            continue

        a1 = create_3d_arrow(
            start=center, direction=axis, length=arrow_length, radius=radius
        )

        a2 = create_3d_arrow(
            start=center, direction=-axis, length=arrow_length, radius=radius
        )

        if a1 is not None:
            arrows.append(a1)

        if a2 is not None:
            arrows.append(a2)

    return arrows


# ---- ROTATIONAL DOF VISUALIZATION ----


def create_rotation_dof_arc(
    center, axis, radius, color=DOF_COLOR, angle_range=(0.0, 2.0 * np.pi)
):
    center = np.asarray(center, dtype=float)

    axis = normalize(axis)

    if axis is None:
        return None

    # --------------------------------------------------------
    # Construct basis perpendicular to axis.
    # --------------------------------------------------------

    reference = np.array([1.0, 0.0, 0.0])

    if abs(np.dot(reference, axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    u = np.cross(axis, reference)

    u = normalize(u)

    if u is None:
        return None

    v = np.cross(axis, u)

    v = normalize(v)

    theta = np.linspace(angle_range[0], angle_range[1], 80)

    points = []

    for t in theta:
        p = center + radius * np.cos(t) * u + radius * np.sin(t) * v

        points.append(p)

    arc = trimesh.load_path(np.asarray(points))

    arc.colors = np.array([color], dtype=np.uint8)

    return arc


def create_rotation_dof_visualization(center, dof_info, axis_length):
    geometries = []

    radius = axis_length * 0.45

    for dof in dof_info["free_dofs"]:
        if dof["type"] != "rotation":
            continue

        axis = dof["axis"]

        arc = create_rotation_dof_arc(center=center, axis=axis, radius=radius)

        if arc is not None:
            geometries.append(arc)

    return geometries


# ---- CONTACT SURFACE ----


def create_contact_surface(poly_2d):

    if len(poly_2d) < 3:
        return None

    poly_2d = np.asarray(poly_2d, dtype=float)

    pts_3d = np.hstack([poly_2d, np.zeros((len(poly_2d), 1))])

    faces = [[0, i, i + 1] for i in range(1, len(pts_3d) - 1)]

    contact_mesh = trimesh.Trimesh(vertices=pts_3d, faces=faces, process=False)

    contact_mesh.visual.face_colors = CONTACT_COLOR

    return contact_mesh


# ---- FREE-DOF GHOSTS ----


def generate_rotation_ghosts(posed_mesh, center, dof_info):
    """
    Generate ghost poses ONLY for explicitly accepted
    rotational DOFs.

    In particular, this will never generate tilt ghosts
    around X/Y for an upright tabletop screw.
    """

    ghosts = []

    rotational_dofs = [d for d in dof_info["free_dofs"] if d["type"] == "rotation"]

    if not rotational_dofs:
        return ghosts

    angles_deg = np.arange(GHOST_ANGLE_STEP_DEG, 360, GHOST_ANGLE_STEP_DEG)

    for dof in rotational_dofs:
        axis = normalize(dof["axis"])

        if axis is None:
            continue

        for angle_deg in angles_deg:
            angle = np.radians(angle_deg)

            T_to_origin = trimesh.transformations.translation_matrix(-center)

            R = trimesh.transformations.rotation_matrix(angle, axis)

            T_back = trimesh.transformations.translation_matrix(center)

            T_rotation = T_back @ R @ T_to_origin

            ghost = posed_mesh.copy().apply_transform(T_rotation)

            ghost.visual.face_colors = GHOST_COLOR

            ghosts.append(ghost)

    return ghosts


# ---- VISUALIZATION ----


def visualize():

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")

        return

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    for filename, poses in data.items():
        mesh_path = os.path.join(ASSET_DIR, filename)

        if not os.path.exists(mesh_path):
            print(f"WARNING: {mesh_path} not found.")

            continue

        # ----------------------------------------------------
        # Load mesh
        # ----------------------------------------------------

        base_mesh = trimesh.load(mesh_path)

        if not isinstance(base_mesh, trimesh.Trimesh):
            base_mesh = base_mesh.dump()[0]

        total_poses = len(poses)

        # ----------------------------------------------------
        # Select representative poses
        # ----------------------------------------------------

        if total_poses > MAX_SAMPLES:
            sample_indices = np.linspace(0, total_poses - 1, MAX_SAMPLES, dtype=int)

            sampled_poses = [poses[i] for i in sample_indices]

        else:
            sampled_poses = poses

        # ====================================================
        # VISUALIZE EACH POSE
        # ====================================================

        for sample_idx, pose in enumerate(sampled_poses):
            transform = np.asarray(pose["transform"], dtype=float)

            poly_2d = np.asarray(pose.get("contact_polygon_2d", []), dtype=float)

            # ------------------------------------------------
            # Apply stable pose
            # ------------------------------------------------

            posed_mesh = base_mesh.copy().apply_transform(transform)

            posed_mesh.visual.face_colors = [200, 200, 220, OBJECT_ALPHA]

            # ------------------------------------------------
            # Object dimensions
            # ------------------------------------------------

            bbox_size = float(np.max(base_mesh.extents))

            axis_len = max(bbox_size * 0.35, 5.0)

            # =================================================
            # WORLD FRAME ONLY
            #
            # The old visualizer showed:
            #
            #     object_axis
            #     world_axis
            #
            # which produced two coordinate frames.
            #
            # We intentionally show only the table/world frame.
            # =================================================

            world_axis = trimesh.creation.axis(
                transform=np.eye(4),
                axis_radius=axis_len * 0.025,
                axis_length=axis_len * 1.2,
            )

            # ------------------------------------------------
            # Center of mass
            # ------------------------------------------------

            com_local = np.asarray(base_mesh.center_mass, dtype=float)

            com_world = trimesh.transformations.transform_points(
                [com_local], transform
            )[0]

            com_marker = trimesh.creation.icosphere(
                subdivisions=2, radius=axis_len * 0.05
            )

            com_marker.apply_translation(com_world)

            com_marker.visual.face_colors = COM_COLOR

            # =================================================
            # PHYSICAL DOF ANALYSIS
            # =================================================

            dof_info = determine_physical_dofs(pose)

            # ------------------------------------------------
            # Translation arrows
            # ------------------------------------------------

            translation_arrows = create_translation_dof_arrows(
                com_world, dof_info, axis_len
            )

            # ------------------------------------------------
            # Rotation arcs
            # ------------------------------------------------

            rotation_arcs = create_rotation_dof_visualization(
                com_world, dof_info, axis_len
            )

            # ------------------------------------------------
            # Ghost meshes
            # ------------------------------------------------

            ghost_meshes = generate_rotation_ghosts(posed_mesh, com_world, dof_info)

            # ------------------------------------------------
            # Contact surface
            # ------------------------------------------------

            contact_surface = create_contact_surface(poly_2d)

            # ------------------------------------------------
            # Table grid
            # ------------------------------------------------

            grid = trimesh.path.creation.grid(
                side=bbox_size * TABLE_GRID_SIZE_FACTOR, count=TABLE_GRID_COUNT
            )

            # =================================================
            # ASSEMBLE SCENE
            # =================================================

            scene_geometries = [posed_mesh, world_axis, com_marker, grid]

            scene_geometries.extend(translation_arrows)

            scene_geometries.extend(rotation_arcs)

            scene_geometries.extend(ghost_meshes)

            # ------------------------------------------------
            # Create 3D Numerical Text Labels for Viewport
            # ------------------------------------------------
            text_labels = create_dof_text_labels(com_world, dof_info, axis_len)
            
            # Append to scene geometry list
            scene_geometries.extend(text_labels)

            if contact_surface is not None:
                scene_geometries.append(contact_surface)

            scene = trimesh.Scene(scene_geometries)

            # =================================================
            # TERMINAL INFORMATION
            # =================================================

            print("\n" + "=" * 75)

            print(f"PART                 : {filename}")

            print(f"TOTAL STABLE CONFIGS : {total_poses}")

            print(f"VISUALIZING SAMPLE   : {sample_idx + 1}/{len(sampled_poses)}")

            print(f"POSE ID              : {pose.get('pose_id', 'N/A')}")

            print(f"STABILITY MARGIN     : {pose.get('stability_margin', 'N/A')} mm")

            print("\nPHYSICAL FREE DOFs:")

            if not dof_info["free_dofs"]:
                print("    None")

            else:
                for dof in dof_info["free_dofs"]:
                    axis = np.round(dof["axis"], 4).tolist()

                    print(f"    {dof['name']} ({dof['type']}) axis={axis}")

            print(f"\nDOF COUNT            : {dof_info['dof_count']}")

            print(
                "\nGreen arrows/arcs/ghosts show only DOFs "
                "compute_stable_poses.py verified as physically "
                "neutral (translation on a planar contact, or a "
                "rotation axis that doesn't change COM height)."
            )

            print("=" * 75)

            caption = (
                f"{filename} | "
                f"Stable pose "
                f"{sample_idx + 1}/"
                f"{len(sampled_poses)} | "
                f"Physical DOF = "
                f"{dof_info['dof_count']}"
            )

            scene.show(caption=caption)


# ---- MAIN ----

if __name__ == "__main__":
    visualize()
