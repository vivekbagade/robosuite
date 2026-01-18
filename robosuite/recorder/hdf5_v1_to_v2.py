import os

import h5py
import numpy as np


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        key_frame = root['/observations/key_frame'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        action = root['/action'][()]

    return qpos, qvel, action, image_dict, key_frame

def rewrite_hdf5_action(dataset_path):
    """
    Rewrites the action dataset in an HDF5 file to be normalized.
    -1 if action < -0.5
     0 if -0.5 <= action <= 0.5
     1 if action > 0.5
    """
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        return

    with h5py.File(dataset_path, 'r+') as root:
        if '/action' not in root:
            print("'/action' dataset not found.")
            return

        action = root['/action'][()]

        # Create a copy to modify
        modified_action = action.copy()

        # Normalize only the last element of the action
        last_action_element = modified_action[:, -1]
        normalized_last_element = np.zeros_like(last_action_element)
        normalized_last_element[last_action_element > 0.1] = 1
        normalized_last_element[last_action_element < -0.1] = -1
        
        # Update the last column in the copied action array
        modified_action[:, -1] = normalized_last_element

        qpos_max = 0
        # Find the point where normalized_last_element switches from 1 to -1
        for i in range(len(normalized_last_element)):
            if i != 0 and normalized_last_element[i] == 0 and normalized_last_element[i-1] == 1:
                qpos_max = i
        
        qpos = root['/observations/qpos'][()]
        modified_qpos = qpos.copy()
        if qpos_max != 0:
            modified_qpos = np.zeros_like(qpos)
            for i in range(qpos_max):
                modified_qpos[i] = qpos[i]

        # Replace the old dataset with the new one
        del root['/action']
        del root['/observations/qpos']
        root.create_dataset('/action', data=modified_action)
        root.create_dataset('/observations/qpos', data=modified_qpos)
        print(f"Action data in {dataset_path} has been normalized and rewritten.")

rewrite_hdf5_action('/media/vivekbagade/Elements/act-data/PickPlaceCan/episodes/1.3.0-sim/episode_52.hdf5')
