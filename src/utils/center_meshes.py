#!/usr/bin/env python3
"""
Center all OBJ meshes at their bounding box center.

Usage:
    python center_meshes.py
"""

import numpy as np
from pathlib import Path


def load_obj(path: str):
    """Load OBJ file, returning vertices, faces, and other lines."""
    vertices = []
    faces = []
    other_lines = []
    
    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('v ') and not stripped.startswith('vt') and not stripped.startswith('vn'):
                parts = stripped.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif stripped.startswith('f '):
                faces.append(line)
            else:
                other_lines.append(line)
    
    return np.array(vertices), faces, other_lines


def save_obj(path: str, vertices: np.ndarray, faces: list, other_lines: list):
    """Save OBJ file with new vertex positions."""
    with open(path, 'w') as f:
        f.write(f"# Centered mesh - origin at bounding box center\n")
        
        for line in other_lines:
            if not line.strip().startswith('#'):
                f.write(line)
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        for face in faces:
            f.write(face)


def center_mesh(input_path: str, output_path: str) -> dict:
    """Center a mesh at its bounding box center."""
    vertices, faces, other_lines = load_obj(input_path)
    
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    
    centered_vertices = vertices - bbox_center
    
    save_obj(output_path, centered_vertices, faces, other_lines)
    
    return {
        'original_center': bbox_center,
        'extents': bbox_max - bbox_min,
        'num_vertices': len(vertices),
    }


def main():
    input_dir = Path('/workspace/MasterThesis/Data/CAD_Models')
    output_dir = Path('/workspace/MasterThesis/Data/CAD_Models_centered')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all .obj files in input directory
    for input_path in input_dir.glob('*.obj'):
        output_path = output_dir / input_path.name  # Keep same filename
        
        info = center_mesh(str(input_path), str(output_path))
        
        print(f"[OK] {input_path.name}")
        print(f"     Shifted by: [{info['original_center'][0]:.4f}, {info['original_center'][1]:.4f}, {info['original_center'][2]:.4f}]")


if __name__ == "__main__":
    main()