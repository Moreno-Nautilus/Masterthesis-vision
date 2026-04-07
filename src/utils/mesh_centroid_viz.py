#!/usr/bin/env python3
"""
Mesh Centroid Visualizer - HTML Output Version

Generates an interactive 3D HTML visualization using Three.js.
No matplotlib/trimesh/scipy needed - just numpy and a browser.

Usage:
    python3 mesh_centroid_viz_html.py /path/to/meshes
    # Opens browser automatically with interactive 3D view
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import webbrowser
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np


@dataclass
class MeshData:
    name: str
    vertices: list  # Flattened [x,y,z,x,y,z,...]
    centroid: list  # [x, y, z]
    bounds_min: list
    bounds_max: list
    extents: list
    num_vertices: int
    num_faces: int
    distance_from_origin: float


def load_stl_binary(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load binary STL file."""
    with open(filepath, 'rb') as f:
        f.read(80)  # Skip header
        num_triangles = struct.unpack('<I', f.read(4))[0]
        
        vertices = []
        faces = []
        
        for i in range(num_triangles):
            f.read(12)  # Skip normal
            v1 = struct.unpack('<fff', f.read(12))
            v2 = struct.unpack('<fff', f.read(12))
            v3 = struct.unpack('<fff', f.read(12))
            f.read(2)  # Skip attribute
            
            base_idx = len(vertices)
            vertices.extend([v1, v2, v3])
            faces.append([base_idx, base_idx + 1, base_idx + 2])
        
        return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_stl_ascii(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ASCII STL file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        current_face = []
        for line in f:
            line = line.strip().lower()
            if line.startswith('vertex'):
                parts = line.split()
                v = [float(parts[1]), float(parts[2]), float(parts[3])]
                current_face.append(len(vertices))
                vertices.append(v)
            elif line.startswith('endfacet'):
                if len(current_face) == 3:
                    faces.append(current_face)
                current_face = []
    
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_stl(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load STL file (auto-detect binary vs ASCII)."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
    
    try:
        if header[:5].decode('ascii').lower() == 'solid':
            with open(filepath, 'r') as f:
                first_lines = f.read(1000)
                if 'facet normal' in first_lines.lower():
                    return load_stl_ascii(filepath)
    except:
        pass
    
    return load_stl_binary(filepath)


def load_obj(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load OBJ file."""
    vertices = []
    faces = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                face_indices = []
                for p in parts:
                    idx = int(p.split('/')[0]) - 1
                    face_indices.append(idx)
                for i in range(1, len(face_indices) - 1):
                    faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
    
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def load_mesh(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh from file based on extension."""
    ext = filepath.suffix.lower()
    
    if ext == '.stl':
        return load_stl(filepath)
    elif ext == '.obj':
        return load_obj(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")


def process_mesh(name: str, path: Path, vertices: np.ndarray, faces: np.ndarray, subsample: int = 3000) -> MeshData:
    """Process mesh and extract visualization data."""
    unique_vertices = np.unique(vertices, axis=0)
    centroid = np.mean(unique_vertices, axis=0)
    bounds_min = np.min(vertices, axis=0)
    bounds_max = np.max(vertices, axis=0)
    extents = bounds_max - bounds_min
    
    # Subsample vertices for visualization (keep it fast)
    if len(vertices) > subsample:
        indices = np.random.choice(len(vertices), subsample, replace=False)
        vis_vertices = vertices[indices]
    else:
        vis_vertices = vertices
    
    return MeshData(
        name=name,
        vertices=vis_vertices.flatten().tolist(),
        centroid=centroid.tolist(),
        bounds_min=bounds_min.tolist(),
        bounds_max=bounds_max.tolist(),
        extents=extents.tolist(),
        num_vertices=len(vertices),
        num_faces=len(faces),
        distance_from_origin=float(np.linalg.norm(centroid)),
    )


def load_all_meshes(mesh_dir: Path) -> list[MeshData]:
    """Load all mesh files from directory."""
    meshes = []
    extensions = ('.stl', '.obj')
    
    mesh_files = []
    for ext in extensions:
        mesh_files.extend(mesh_dir.glob(f"*{ext}"))
        mesh_files.extend(mesh_dir.glob(f"*{ext.upper()}"))
        mesh_files.extend(mesh_dir.glob(f"**/*{ext}"))
        mesh_files.extend(mesh_dir.glob(f"**/*{ext.upper()}"))
    
    mesh_files = sorted(set(mesh_files))
    
    print(f"\nFound {len(mesh_files)} mesh files\n")
    
    for path in mesh_files:
        try:
            vertices, faces = load_mesh(path)
            info = process_mesh(path.stem, path, vertices, faces)
            meshes.append(info)
            print(f"  ✅ {path.name}: centroid=[{info.centroid[0]:.2f}, {info.centroid[1]:.2f}, {info.centroid[2]:.2f}]")
        except Exception as e:
            print(f"  ❌ {path.name}: {e}")
    
    return meshes


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mesh Centroid Visualizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e; 
            color: #eee;
            overflow: hidden;
        }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            max-width: 350px;
            max-height: 90vh;
            overflow-y: auto;
            z-index: 100;
        }
        #info h1 { font-size: 18px; margin-bottom: 10px; color: #4fc3f7; }
        #info h2 { font-size: 14px; margin: 15px 0 8px 0; color: #81c784; }
        .mesh-btn {
            display: block;
            width: 100%;
            padding: 8px 12px;
            margin: 4px 0;
            background: #2d2d44;
            border: 1px solid #444;
            border-radius: 4px;
            color: #fff;
            cursor: pointer;
            text-align: left;
            font-size: 13px;
        }
        .mesh-btn:hover { background: #3d3d5c; }
        .mesh-btn.active { background: #4fc3f7; color: #000; }
        #details {
            margin-top: 15px;
            padding: 10px;
            background: #2d2d44;
            border-radius: 4px;
            font-size: 12px;
            line-height: 1.6;
        }
        #details .label { color: #888; }
        #details .value { color: #4fc3f7; font-family: monospace; }
        #details .warning { color: #ffb74d; }
        #details .good { color: #81c784; }
        #controls {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0,0,0,0.8);
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 12px;
            color: #888;
        }
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border-radius: 8px;
            font-size: 12px;
        }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 16px; height: 16px; border-radius: 50%; margin-right: 8px; }
    </style>
</head>
<body>
    <div id="container"></div>
    
    <div id="info">
        <h1>🔍 Mesh Centroid Visualizer</h1>
        <h2>Select Mesh:</h2>
        <div id="mesh-list"></div>
        <div id="details"></div>
    </div>
    
    <div id="legend">
        <div class="legend-item"><div class="legend-color" style="background: #ff4444;"></div> Centroid</div>
        <div class="legend-item"><div class="legend-color" style="background: #000;"></div> Origin (0,0,0)</div>
        <div class="legend-item"><div class="legend-color" style="background: #ff0000; width: 20px; height: 3px; border-radius: 0;"></div> X Axis</div>
        <div class="legend-item"><div class="legend-color" style="background: #00ff00; width: 20px; height: 3px; border-radius: 0;"></div> Y Axis</div>
        <div class="legend-item"><div class="legend-color" style="background: #0000ff; width: 20px; height: 3px; border-radius: 0;"></div> Z Axis</div>
    </div>
    
    <div id="controls">
        🖱️ Left drag: Rotate | Right drag: Pan | Scroll: Zoom | Click mesh name to focus
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Mesh data injected by Python
        const MESH_DATA = __MESH_DATA__;
        
        // Three.js setup
        let scene, camera, renderer, controls;
        let meshGroups = {};
        let currentMesh = null;
        
        const COLORS = [
            0x4fc3f7, 0x81c784, 0xffb74d, 0xf06292, 
            0xba68c8, 0x4dd0e1, 0xaed581, 0xff8a65
        ];
        
        function init() {
            // Scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);
            
            // Camera
            camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
            camera.position.set(50, 50, 50);
            
            // Renderer
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(50, 100, 50);
            scene.add(directionalLight);
            
            // Grid
            const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x333333);
            scene.add(gridHelper);
            
            // Origin marker
            const originGeom = new THREE.SphereGeometry(1, 16, 16);
            const originMat = new THREE.MeshBasicMaterial({ color: 0x000000 });
            const originMarker = new THREE.Mesh(originGeom, originMat);
            scene.add(originMarker);
            
            // Origin axes
            const axisLength = 20;
            const axisX = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), axisLength, 0xff0000, 2, 1);
            const axisY = new THREE.ArrowHelper(new THREE.Vector3(0,1,0), new THREE.Vector3(0,0,0), axisLength, 0x00ff00, 2, 1);
            const axisZ = new THREE.ArrowHelper(new THREE.Vector3(0,0,1), new THREE.Vector3(0,0,0), axisLength, 0x0000ff, 2, 1);
            scene.add(axisX, axisY, axisZ);
            
            // Create mesh visualizations
            createMeshVisualizations();
            
            // Controls (simple orbit)
            setupControls();
            
            // UI
            createMeshButtons();
            
            // Auto-select first mesh
            if (MESH_DATA.length > 0) {
                selectMesh(MESH_DATA[0].name);
            }
            
            // Animation loop
            animate();
            
            // Resize handler
            window.addEventListener('resize', onWindowResize);
        }
        
        function createMeshVisualizations() {
            MESH_DATA.forEach((mesh, idx) => {
                const group = new THREE.Group();
                group.name = mesh.name;
                
                const color = COLORS[idx % COLORS.length];
                
                // Point cloud for mesh vertices
                const geometry = new THREE.BufferGeometry();
                const vertices = new Float32Array(mesh.vertices);
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                
                const material = new THREE.PointsMaterial({ 
                    color: color, 
                    size: 0.5,
                    transparent: true,
                    opacity: 0.6
                });
                const points = new THREE.Points(geometry, material);
                group.add(points);
                
                // Centroid marker (red sphere)
                const centroidGeom = new THREE.SphereGeometry(Math.max(...mesh.extents) * 0.03, 16, 16);
                const centroidMat = new THREE.MeshBasicMaterial({ color: 0xff4444 });
                const centroidMarker = new THREE.Mesh(centroidGeom, centroidMat);
                centroidMarker.position.set(...mesh.centroid);
                group.add(centroidMarker);
                
                // Axes at centroid
                const axLen = Math.max(...mesh.extents) * 0.2;
                const centroidPos = new THREE.Vector3(...mesh.centroid);
                const axX = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), centroidPos, axLen, 0xff0000, axLen*0.1, axLen*0.05);
                const axY = new THREE.ArrowHelper(new THREE.Vector3(0,1,0), centroidPos, axLen, 0x00ff00, axLen*0.1, axLen*0.05);
                const axZ = new THREE.ArrowHelper(new THREE.Vector3(0,0,1), centroidPos, axLen, 0x0000ff, axLen*0.1, axLen*0.05);
                group.add(axX, axY, axZ);
                
                // Bounding box
                const boxGeom = new THREE.BoxGeometry(...mesh.extents);
                const boxMat = new THREE.MeshBasicMaterial({ 
                    color: color, 
                    wireframe: true,
                    transparent: true,
                    opacity: 0.3
                });
                const box = new THREE.Mesh(boxGeom, boxMat);
                const boxCenter = mesh.bounds_min.map((min, i) => min + mesh.extents[i] / 2);
                box.position.set(...boxCenter);
                group.add(box);
                
                // Line from origin to centroid
                const lineGeom = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(0, 0, 0),
                    new THREE.Vector3(...mesh.centroid)
                ]);
                const lineMat = new THREE.LineDashedMaterial({ 
                    color: 0xffff00, 
                    dashSize: 2, 
                    gapSize: 1,
                    transparent: true,
                    opacity: 0.5
                });
                const line = new THREE.Line(lineGeom, lineMat);
                line.computeLineDistances();
                group.add(line);
                
                group.visible = false;
                scene.add(group);
                meshGroups[mesh.name] = group;
            });
        }
        
        function selectMesh(name) {
            // Hide all meshes
            Object.values(meshGroups).forEach(g => g.visible = false);
            
            // Show selected
            if (meshGroups[name]) {
                meshGroups[name].visible = true;
                currentMesh = MESH_DATA.find(m => m.name === name);
                
                // Update camera to look at mesh
                const center = currentMesh.centroid;
                const maxExtent = Math.max(...currentMesh.extents);
                camera.position.set(
                    center[0] + maxExtent * 1.5,
                    center[1] + maxExtent * 1.5,
                    center[2] + maxExtent * 1.5
                );
                camera.lookAt(new THREE.Vector3(...center));
                
                // Update UI
                updateDetails(currentMesh);
                updateButtons(name);
            }
        }
        
        function createMeshButtons() {
            const container = document.getElementById('mesh-list');
            MESH_DATA.forEach((mesh, idx) => {
                const btn = document.createElement('button');
                btn.className = 'mesh-btn';
                btn.textContent = mesh.name;
                btn.onclick = () => selectMesh(mesh.name);
                btn.style.borderLeftColor = '#' + COLORS[idx % COLORS.length].toString(16).padStart(6, '0');
                btn.style.borderLeftWidth = '4px';
                container.appendChild(btn);
            });
        }
        
        function updateButtons(activeName) {
            document.querySelectorAll('.mesh-btn').forEach(btn => {
                btn.classList.toggle('active', btn.textContent === activeName);
            });
        }
        
        function updateDetails(mesh) {
            const isNearOrigin = mesh.distance_from_origin < 1;
            const unitGuess = Math.max(...mesh.extents) > 100 ? 'mm' : 'm';
            
            document.getElementById('details').innerHTML = `
                <div><span class="label">Centroid:</span></div>
                <div><span class="value">X: ${mesh.centroid[0].toFixed(4)}</span></div>
                <div><span class="value">Y: ${mesh.centroid[1].toFixed(4)}</span></div>
                <div><span class="value">Z: ${mesh.centroid[2].toFixed(4)}</span></div>
                
                <div style="margin-top: 10px;"><span class="label">Bounding Box:</span></div>
                <div><span class="value">${mesh.extents[0].toFixed(2)} × ${mesh.extents[1].toFixed(2)} × ${mesh.extents[2].toFixed(2)}</span></div>
                <div><span class="label">Likely units: </span><span class="value">${unitGuess}</span></div>
                
                <div style="margin-top: 10px;"><span class="label">Distance from origin:</span></div>
                <div class="${isNearOrigin ? 'good' : 'warning'}">${mesh.distance_from_origin.toFixed(4)} ${unitGuess}</div>
                
                <div style="margin-top: 10px;"><span class="label">Mesh stats:</span></div>
                <div><span class="value">${mesh.num_vertices.toLocaleString()} vertices, ${mesh.num_faces.toLocaleString()} faces</span></div>
                
                ${!isNearOrigin ? `
                <div style="margin-top: 15px; padding: 10px; background: #3d2d2d; border-radius: 4px;">
                    <div class="warning">⚠️ Centroid far from origin!</div>
                    <div style="margin-top: 5px; font-size: 11px; color: #aaa;">
                        Suggested offsets for grasp_and_insert_demo.py:
                    </div>
                    <div style="font-family: monospace; font-size: 10px; margin-top: 5px; color: #4fc3f7;">
                        -p centroid_offset_x:=${(-mesh.centroid[0]).toFixed(6)}<br>
                        -p centroid_offset_y:=${(-mesh.centroid[1]).toFixed(6)}<br>
                        -p centroid_offset_z:=${(-mesh.centroid[2]).toFixed(6)}
                    </div>
                </div>
                ` : `
                <div style="margin-top: 15px; padding: 10px; background: #2d3d2d; border-radius: 4px;">
                    <div class="good">✅ Centroid near origin - good for FoundationPose!</div>
                </div>
                `}
            `;
        }
        
        function setupControls() {
            let isMouseDown = false;
            let isRightMouseDown = false;
            let prevMouseX = 0, prevMouseY = 0;
            
            const container = document.getElementById('container');
            
            container.addEventListener('mousedown', (e) => {
                if (e.button === 0) isMouseDown = true;
                if (e.button === 2) isRightMouseDown = true;
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            });
            
            container.addEventListener('mouseup', () => {
                isMouseDown = false;
                isRightMouseDown = false;
            });
            
            container.addEventListener('mousemove', (e) => {
                if (!isMouseDown && !isRightMouseDown) return;
                
                const deltaX = e.clientX - prevMouseX;
                const deltaY = e.clientY - prevMouseY;
                
                if (isMouseDown) {
                    // Orbit
                    const target = currentMesh ? new THREE.Vector3(...currentMesh.centroid) : new THREE.Vector3(0, 0, 0);
                    const offset = camera.position.clone().sub(target);
                    
                    const spherical = new THREE.Spherical();
                    spherical.setFromVector3(offset);
                    spherical.theta -= deltaX * 0.01;
                    spherical.phi -= deltaY * 0.01;
                    spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
                    
                    offset.setFromSpherical(spherical);
                    camera.position.copy(target).add(offset);
                    camera.lookAt(target);
                }
                
                if (isRightMouseDown) {
                    // Pan
                    const panSpeed = 0.5;
                    camera.position.x -= deltaX * panSpeed;
                    camera.position.y += deltaY * panSpeed;
                }
                
                prevMouseX = e.clientX;
                prevMouseY = e.clientY;
            });
            
            container.addEventListener('wheel', (e) => {
                e.preventDefault();
                const target = currentMesh ? new THREE.Vector3(...currentMesh.centroid) : new THREE.Vector3(0, 0, 0);
                const direction = camera.position.clone().sub(target).normalize();
                const distance = camera.position.distanceTo(target);
                const newDistance = distance * (1 + e.deltaY * 0.001);
                camera.position.copy(target).add(direction.multiplyScalar(newDistance));
            });
            
            container.addEventListener('contextmenu', (e) => e.preventDefault());
        }
        
        function onWindowResize() {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }
        
        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }
        
        init();
    </script>
</body>
</html>
'''


def generate_html(meshes: list[MeshData], output_path: Path) -> None:
    """Generate HTML visualization file."""
    mesh_data_json = json.dumps([{
        'name': m.name,
        'vertices': m.vertices,
        'centroid': m.centroid,
        'bounds_min': m.bounds_min,
        'bounds_max': m.bounds_max,
        'extents': m.extents,
        'num_vertices': m.num_vertices,
        'num_faces': m.num_faces,
        'distance_from_origin': m.distance_from_origin,
    } for m in meshes])
    
    html_content = HTML_TEMPLATE.replace('__MESH_DATA__', mesh_data_json)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive 3D mesh centroid visualization")
    parser.add_argument("mesh_dir", help="Directory containing mesh files")
    parser.add_argument("-o", "--output", help="Output HTML file (default: opens temp file in browser)")
    
    args = parser.parse_args()
    
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()
    
    if not mesh_dir.exists():
        print(f"ERROR: Directory not found: {mesh_dir}")
        sys.exit(1)
    
    print(f"🔍 Scanning: {mesh_dir}")
    meshes = load_all_meshes(mesh_dir)
    
    if not meshes:
        print("\n❌ No meshes loaded")
        sys.exit(1)
    
    # Generate HTML
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(tempfile.gettempdir()) / "mesh_centroids.html"
    
    generate_html(meshes, output_path)
    
    # Open in browser
    print(f"🌐 Opening in browser...")
    webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()