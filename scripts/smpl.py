import numpy as np

class SMPLModel:
    
    def __init__(self, model_path):
        with np.load(model_path) as data:
            self.v_template = data['vertices_template']
            self.shapedirs = data['shape_blend_shapes']
            self.posedirs = data['pose_blend_shapes']
            self.weights = data['weights']
            self.J_regressor = data['joint_regressor']
            self.kintree_table = data['kinematic_tree']
            self.faces = data['face_indices'] - 1 # Convert back to 0-indexed

    def rodrigues(self, r):
        """Rotates a vector r into a rotation matrix."""
        theta = np.linalg.norm(r)
        if theta < 1e-8:
            return np.eye(3)
        r = r / theta
        K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.matmul(K, K)

    def __call__(self, beta, pose):
        """
        beta: (10,) shape parameters
        pose: (24, 3) pose parameters
        """

        # 1. Shape Blend Shapes
        v_shaped = self.v_template + np.tensordot(self.shapedirs, beta, axes=([2], [0]))

        # 2. Joint Locations
        J = np.matmul(self.J_regressor, v_shaped)

        # 3. Pose Blend Shapes
        pose_matrix = [self.rodrigues(p) for p in pose]

        # Ignore global rotation for the pose blend shape calculation
        pose_map = np.concatenate([p - np.eye(3) for p in pose_matrix[1:]]).flatten()
        v_posed = v_shaped + np.tensordot(self.posedirs, pose_map, axes=([2], [0]))

        # 4. Forward Kinematics
        global_transforms = np.zeros((24, 4, 4))
        parents = self.kintree_table[0].astype(int)
        
        # Root rotation
        global_transforms[0, :3, :3] = pose_matrix[0]
        global_transforms[0, :3, 3] = J[0]
        global_transforms[0, 3, 3] = 1

        for i in range(1, 24):
            parent = parents[i]
            local_rel_transform = np.eye(4)
            local_rel_transform[:3, :3] = pose_matrix[i]
            local_rel_transform[:3, 3] = J[i] - J[parent]
            global_transforms[i] = np.matmul(global_transforms[parent], local_rel_transform)

        # Remove joint rest position from transforms
        for i in range(24):
            j_rest = np.eye(4)
            j_rest[:3, 3] = -J[i]
            global_transforms[i] = np.matmul(global_transforms[i], j_rest)

        # 5. Linear Blend Skinning (LBS)
        T = np.tensordot(self.weights, global_transforms, axes=([1], [0]))
        v_homo = np.hstack((v_posed, np.ones((v_posed.shape[0], 1))))
        v_final = np.matmul(T, v_homo[:, :, np.newaxis])[:, :3, 0]

        return v_final, self.faces

# --- Example Usage ---
if __name__ == "__main__":

    # Path to the .npz file created by preprocess.py
    model = SMPLModel('../models/smpl_female.npz')

    # Parameters: Zero (Neutral T-Pose and Neutral Shape)
    betas = np.zeros(10) 
    pose = np.zeros((24, 3)) 

    # Generate 3D Mesh
    verts, faces = model(betas, pose)

    # Save as .obj for viewing in MeshLab
    with open('output.obj', 'w') as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print("Mesh generated: output.obj")