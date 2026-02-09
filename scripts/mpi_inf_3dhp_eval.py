import json
import numpy as np
import h5py
import trimesh
import os

# Metrics

def compute_mpjpe(pred, gt):
    """ Mean Per Joint Position Error (Euclidean distance) """
    # pred, gt: (N, 17, 3)
    return np.mean(np.linalg.norm(pred - gt, axis=2))

def compute_pa_mpjpe(pred, gt):
    """ Procrustes Aligned MPJPE (Rigid alignment) """
    pred_aligned = np.zeros_like(pred)
    for i in range(len(pred)):
        p = pred[i]
        g = gt[i]
        # Procrustes Analysis
        u, s, vt = np.linalg.svd(g.T @ p)
        R = u @ vt
        if np.linalg.det(R) < 0:
            vt[2, :] *= -1
            R = u @ vt
        pred_aligned[i] = p @ R.T
    return compute_mpjpe(pred_aligned, gt)

def compute_pck(errors, threshold=150.0):
    """ Percentage of Correct Keypoints < threshold """
    return np.mean(errors < threshold) * 100.0

def compute_auc(errors, range_threshold=150.0, step=5.0):
    """ Area Under Curve (0 to 150mm) """
    thresholds = np.arange(0, range_threshold + 1, step)
    pck_values = [compute_pck(errors, t) for t in thresholds]
    return np.mean(pck_values)

# Mappings
MPI_INF_3DHP_TO_H36M = [10, 8, 14, 15, 16, 11, 12, 13, 4, 5, 6, 1, 2, 3, 0, 7, 9]
ROOT_IDX = 14 

# Paths
base_path = "." 
regressor_path = os.path.join(base_path, "data/J_regressor_h36m.npy")

test_sets = ["TS1", "TS2", "TS3", "TS4", "TS5", "TS6"]

annot_paths = [os.path.join(base_path, f"data/mpi_inf_3dhp_test_set/mpi_inf_3dhp_test_set/{ts}/annot_data.mat") for ts in test_sets]
result_paths = [os.path.join(base_path, f"proposal_results/{ts}/debug.json") for ts in test_sets]

# Load Regressor
h36m_joint_regressor = np.load(regressor_path)

# Accumulators
total_mpjpe = []
total_pa_mpjpe = []
total_joint_errors = [] # For PCK/AUC

print(f"{'='*60}")
print(f"STARTING EVALUATION ON {len(test_sets)} SEQUENCES")
print(f"{'='*60}")

for video_idx, (annot_path, result_path) in enumerate(zip(annot_paths, result_paths)):
    
    seq_name = test_sets[video_idx]
    print(f"Processing {seq_name}...", end=" ")

    # Load ground truth
    with h5py.File(annot_path, 'r') as fp:
        gt_joints_all = fp['annot3'][()]      # (N, 17, 3)
        valid_frames = fp['valid_frame'][()]  # (N, 1)

    gt_joints_all = gt_joints_all.reshape(-1, 17, 3)
    valid_frames = valid_frames.flatten()

    # Load pipeline results
    with open(result_path, "r") as f:
        data = json.load(f)
    
    pred_seq = []
    gt_seq = []
    current_valid_mask = []

    # breakpoint()

    # --- B. PROCESS SEQUENCE ---
    for frame_idx, frame_data in enumerate(data["frames"]):

        if valid_frames[frame_idx] == 0:
            continue

        # Prediction
        mesh_path = frame_data["mesh_path"]
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(mesh_path)
            
        mesh = trimesh.load(mesh_path, process=False)
        pred_pose = np.matmul(h36m_joint_regressor, mesh.vertices)
        pred_pose = pred_pose[MPI_INF_3DHP_TO_H36M]
        pred_pose_mm = pred_pose * 1000.0 # Convert to MM
        
        # Ground Truth
        gt_pose_mm = gt_joints_all[frame_idx]
        
        # Root Centering (Critical for MPJPE)
        pred_centered = pred_pose_mm - pred_pose_mm[ROOT_IDX]
        gt_centered = gt_pose_mm - gt_pose_mm[ROOT_IDX]
        
        pred_seq.append(pred_centered)
        gt_seq.append(gt_centered)

    pred_seq = np.array(pred_seq)
    gt_seq = np.array(gt_seq)

    # MPJPE
    curr_errors = np.linalg.norm(pred_seq - gt_seq, axis=2)
    total_mpjpe.append(np.mean(curr_errors))
    total_joint_errors.extend(curr_errors.flatten()) # For PCK

    # PA-MPJPE
    total_pa_mpjpe.append(compute_pa_mpjpe(pred_seq, gt_seq))

# Results
print("\n" + "="*60)
print(" FINAL EVALUATION REPORT ")
print("="*60)

all_errors = np.array(total_joint_errors)

print(f"{'METRIC':<25} | {'VALUE':<10} | {'UNIT'}")
print("-" * 50)
print(f"{'MPJPE':<25} | {np.mean(total_mpjpe):.2f}       | mm")
print(f"{'PA-MPJPE':<25} | {np.mean(total_pa_mpjpe):.2f}       | mm")
print(f"{'3DPCK @ 150mm':<25} | {compute_pck(all_errors, 150):.2f}       | %")
print(f"{'AUC':<25} | {compute_auc(all_errors):.2f}       | -")
print("-" * 50)