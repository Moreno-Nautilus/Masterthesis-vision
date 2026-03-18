import trimesh
import numpy as np
import sys

def inspect_mesh(path):
    print("\n" + "="*50)
    print(f"Inspecting: {path}")
    print("="*50)

    mesh = trimesh.load(path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        print("⚠️ Not a valid mesh")
        return

    bounds = mesh.bounds
    extent = mesh.extents
    center = mesh.bounding_box.centroid
    com = mesh.center_mass

    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces:    {len(mesh.faces)}")

    print("\n--- BOUNDS ---")
    print(f"min: {bounds[0]}")
    print(f"max: {bounds[1]}")

    print("\n--- SIZE (meters!) ---")
    print(f"extent xyz: {extent}")

    print("\n--- ORIGIN ANALYSIS ---")
    print(f"Bounding box center: {center}")
    print(f"Center of mass:      {com}")

    origin = np.array([0, 0, 0])
    dists = np.linalg.norm(mesh.vertices - origin, axis=1)

    print(f"\nDistance of origin to closest vertex: {dists.min():.4f} m")

    if np.all(bounds[0] <= 0) and np.all(bounds[1] >= 0):
        print("✅ Origin is INSIDE the mesh bounds")
    else:
        print("❌ Origin is OUTSIDE the mesh bounds")

    # Heuristic checks
    print("\n--- INTERPRETATION ---")

    if np.allclose(center, [0,0,0], atol=1e-3):
        print("→ Origin is roughly CENTERED")
    elif abs(bounds[0][2]) < 1e-3:
        print("→ Origin is likely on the BOTTOM (z=0)")
    else:
        print("→ Origin is OFFSET / arbitrary")

    print("="*50)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_cad.py path_to_mesh")
    else:
        inspect_mesh(sys.argv[1])