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

# Script was obtained from https://github.com/chongyi-zheng/SMPLpp
# We did some modifications to work with Python 3.

import sys
import os
import numpy as np
import pickle as pkl
import json

def main(args):
    gender = args[1]
    raw_model_path = args[2]
    save_dir = args[3]

    output_names = {
        'female': ('smpl_female.npz', 'smpl_female.json'),
        'male': ('smpl_male.npz', 'smpl_male.json')
    }

    if gender not in output_names:
        raise SystemError("Gender must be 'male' or 'female'")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load with latin1 and convert chumpy objects to numpy
    with open(raw_model_path, 'rb') as f:
        data = pkl.load(f, encoding='latin1')

    # Helper to strip chumpy/scipy wrappers
    def to_array(x):
        if hasattr(x, 'toarray'): # for scipy sparse matrices
            return np.array(x.toarray())
        return np.array(x)

    model_data = {
        'vertices_template': to_array(data['v_template']),
        'face_indices': to_array(data['f']).astype(np.int32) + 1,
        'weights': to_array(data['weights']),
        'shape_blend_shapes': to_array(data['shapedirs']),
        'pose_blend_shapes': to_array(data['posedirs']),
        'joint_regressor': to_array(data['J_regressor']),
        'kinematic_tree': to_array(data['kintree_table']).astype(np.int32)
    }

    # Save NPZ
    npz_path = os.path.join(save_dir, output_names[gender][0])
    np.savez(npz_path, **model_data)

    # Save JSON (Convert to list for serialization)
    json_path = os.path.join(save_dir, output_names[gender][1])
    model_json = {k: v.tolist() for k, v in model_data.items()}
    
    with open(json_path, 'w') as f: # Use 'w' instead of 'wb+'
        json.dump(model_json, f, indent=4)

    print(f'Successfully processed model to: {os.path.abspath(save_dir)}')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('USAGE: python3 preprocess.py <gender> <path-to-pkl> <save-dir>')
        sys.exit(1)
    main(sys.argv)