#!/usr/bin/env python3
"""
render.py
---------------
View SMPL mesh sequence in 3D space.

Usage:
    python animate_mesh.py --output-dir build/output
"""

import argparse
import json
import os
import numpy as np
import open3d as o3d


def load_obj(path: str) -> tuple:
    """Load OBJ mesh file, returns (vertices, faces)."""
    vertices = []
    faces = []
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                # OBJ uses 1-based indexing
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(face)
    
    return np.array(vertices), np.array(faces)


class MeshPlayer:
    def __init__(self, frames_data, output_dir, geometry):
        self.frames_data = frames_data
        self.output_dir = output_dir
        self.mesh_geo = geometry
        self.idx = 0
        self.total = len(frames_data)
        self.update_mesh(0)

    def update_mesh(self, new_idx):
        """Loads the mesh for the specific index from disk."""
        # Clamp index
        if new_idx < 0: new_idx = 0
        if new_idx >= self.total: new_idx = self.total - 1
        
        self.idx = new_idx
        fd = self.frames_data[self.idx]
        
        # Path to mesh
        mesh_path = os.path.join("../build", fd['mesh_path'])
   
        # Load Data
        vertices, faces = load_obj(mesh_path)
        global_t = np.array(fd['globalT'])
        
        # Update Open3D Geometry in-place
        self.mesh_geo.vertices = o3d.utility.Vector3dVector(vertices + global_t)
        self.mesh_geo.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh_geo.compute_vertex_normals()
        
        print(f"\rFrame {fd['frame']:5d} ({self.idx + 1}/{self.total})", end='', flush=True)

    def next_frame(self, vis):
        self.update_mesh(self.idx + 1)
        vis.update_geometry(self.mesh_geo)
        return False 

    def prev_frame(self, vis):
        self.update_mesh(self.idx - 1)
        vis.update_geometry(self.mesh_geo)
        return False


def main():
    parser = argparse.ArgumentParser(description='Navigate SMPL mesh sequence')
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Path to pipeline output directory'
    )
    args = parser.parse_args()
    
    # Load Config
    debug_path = os.path.join(args.output_dir, 'debug.json')
    with open(debug_path, 'r') as f:
        debug_data = json.load(f)
    frames_data = debug_data['frames']

    # Setup Visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="SMPL Navigator (Left/Right Arrows)", width=1280, height=720)
    
    # Setup Scene
    # Create a dummy mesh to initialize
    mesh = o3d.geometry.TriangleMesh()
    mesh.paint_uniform_color([0.7, 0.75, 0.9])
    
    # Initialize Player
    player = MeshPlayer(frames_data, args.output_dir, mesh)
    
    vis.add_geometry(mesh)
    
    # Environment (Grid/Ground)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)
    
    ground = o3d.geometry.TriangleMesh.create_box(width=2.0, height=0.01, depth=2.0)
    ground.translate([-1.0, 0, -1.0])
    ground.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(ground)

    # Render Options
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0.1, 0.1, 0.15])
    render_opt.light_on = True

    # Camera Setup
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])

    # Register Callbacks for arrows
    vis.register_key_callback(262, player.next_frame) 
    vis.register_key_callback(263, player.prev_frame)

    print("\nControls:")
    print("  [Right Arrow] : Next Frame")
    print("  [Left Arrow]  : Previous Frame")
    print("  [Q]           : Quit")
    
    # 5. Run
    vis.run()
    vis.destroy_window()
    print()

if __name__ == '__main__':
    main()