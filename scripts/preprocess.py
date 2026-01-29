# Copyright 2018 Chongyi Zheng. All rights reserved.
#
# This software implements a 3D human skinning model, SMPL, with tensorflow
# and numpy.
# For more detail, see the paper - SMPL: A Skinned Multi-Person Linear Model -
# published by Max Planck Institute for Intelligent Systems on SIGGRAPH ASIA 2015.
#
# Here we provide the software for research purposes only.
# More information about SMPL is available on http://smpl.is.tue.mpg.
#
# ============================= preprocess.py =================================
# File Description:
#
# This file loads the models downloaded from the official SMPL website, grab
# data and write them in to numpy and json format.
#
# =============================================================================

# Script was obtained from https://github.com/chongyi-zheng/SMPLpp with some modifications

import sys
import os
import numpy as np
import pickle as pkl
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Preprocess SMPL models into JSON.')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the raw SMPL model (pickle)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save the processed output')
    parser.add_argument('--output_name', type=str, required=True,
                        help='Name of the output JSON file (e.g., smpl_model.json)')
    parser.add_argument('--gmm', type=str, required=True,
                        help='Path to the GMM prior (pickle)')
    parser.add_argument('--openpose_joint_regressor', type=str, required=True,
                        help='Path to the OpenPose joint regressor (.npy)')
    
    # Optional arguments
    parser.add_argument('--capsule_regressor_path', type=str, default=None,
                        help='Path to the capsule regressors .npz file (Optional)')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Helper to strip chumpy/scipy wrappers and convert to numpy
    def to_array(x):
        if hasattr(x, 'toarray'): # for scipy sparse matrices
            return np.array(x.toarray())
        return np.array(x)

    # Load SMPL Model
    print(f"Loading SMPL model from: {args.model_path}")
    with open(args.model_path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')

    model_data = {
        'vertices_template': to_array(data['v_template']),
        'face_indices': to_array(data['f']).astype(np.int32) + 1, # +1 for 1-based indexing
        'weights': to_array(data['weights']),
        'shape_blend_shapes': to_array(data['shapedirs']),
        'pose_blend_shapes': to_array(data['posedirs']),
        'joint_regressor': to_array(data['J_regressor']),
        'kinematic_tree': to_array(data['kintree_table']).astype(np.int32)
    }

    # Load GMM
    print(f"Loading GMM from: {args.gmm}")
    with open(args.gmm, 'rb') as fp:
        gmm_data = pkl.load(fp, encoding='latin1')
        model_data['gmm_means'] = to_array(gmm_data['means'])
        model_data['gmm_covars'] = to_array(gmm_data['covars'])
        model_data['gmm_weights'] = to_array(gmm_data['weights'])

    # Load OpenPose Joint Regressor 
    print(f"Loading OpenPose joint regressor from: {args.openpose_joint_regressor}")
    with open(args.openpose_joint_regressor, 'rb') as fp:
        model_data["openpose_joint_regressor"] = np.load(fp)
    

    # Load Capsule Regressors (Optional)
    if args.capsule_regressor_path:
        print(f"Loading Capsule Regressors from: {args.capsule_regressor_path}")
        reg_data = np.load(args.capsule_regressor_path)
        for key in reg_data.files:
            model_data[f'capsule_{key}'] = to_array(reg_data[key])
    else:
        print("No Capsule Regressor path provided. Skipping capsule data.")

    # Save JSON
    json_path = os.path.join(args.save_dir, args.output_name)
    print(f"Saving JSON to: {json_path}")
    
    model_json = {}
    for k, v in model_data.items():
        if isinstance(v, np.ndarray):
            model_json[k] = v.tolist()
        else:
            model_json[k] = v

    with open(json_path, 'w') as f: 
        json.dump(model_json, f, indent=4)

    print(f'Successfully processed.')

if __name__ == '__main__':
    main()