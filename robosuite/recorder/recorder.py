import h5py
import time
import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import rlds

cam_height = 256
cam_width = 256
episode_len = 800

class Recorder:
    def __init__(self, cameras, task) -> None:
        self.data_dir = "data"
        self.data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        self.cameras = cameras
        self.task = task

        for cam_name in cameras:
            self.data_dict[f'/observations/images/{cam_name}'] = []
        

    def record(self, obs, action) -> None:
        qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
        self.data_dict['/observations/qpos'].append(np.concatenate((qpos, obs['grasp'])))
        self.data_dict['/observations/qvel'].append(np.concatenate((obs['robot0_joint_vel'], obs['grasp'])))
        self.data_dict['/action'].append(action)
        for cam_name in self.cameras:
            self.data_dict[f'/observations/images/{cam_name}'].append(obs[cam_name + "_image"])

    def save(self) -> None:
        max_timesteps = len(self.data_dict['/observations/qpos'])
        if max_timesteps < 10:
            print('Not enough steps to save episode')
            return
        if max_timesteps > episode_len:
            print('recording longer than expected, skipping save')
            return

        # padding to episode_len        
        pad_len = episode_len - max_timesteps
        self.data_dict['/observations/qpos'] = np.pad(self.data_dict['/observations/qpos'], ((0, pad_len), (0, 0)), mode='constant')
        self.data_dict['/observations/qvel'] = np.pad(self.data_dict['/observations/qvel'], ((0, pad_len), (0, 0)), mode='constant')
        self.data_dict['/action'] = np.pad(self.data_dict['/action'], ((0, pad_len), (0, 0)), mode='constant')
        for cam_name in self.cameras:
            self.data_dict[f'/observations/images/{cam_name}'] = np.pad(self.data_dict[f'/observations/images/{cam_name}'], ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')

        # create data dir if it doesn't exist
        self.data_dir = os.path.join(self.data_dir, self.task)
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))])
        dataset_path = os.path.join(self.data_dir, f'episode_{idx}')
        print(f"Saving episode to {dataset_path}")
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.cameras:
                _ = image.create_dataset(cam_name, (episode_len, cam_height, cam_width, 3), dtype='uint8',
                                        chunks=(1, cam_height, cam_width, 3), )
            qpos = obs.create_dataset('qpos', (episode_len, 8))
            qvel = obs.create_dataset('qvel', (episode_len, 8))
            # image = obs.create_dataset("image", (episode_len, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (episode_len, 7))
            
            for name, array in self.data_dict.items():
                root[name][...] = array

class RobosuiteRecorder:
    def __init__(self, cameras, task, episode_len, save_dir) -> None:
        self.episode = np.array([])
        self.cameras = cameras
        self.task = task
        self.episode_len = episode_len
        self.save_dir = save_dir

    def record(self, obs, action) -> None:
        data_dict = {}

        # [joint pos cos][joint pos sin][joint vel][EEF XYZ quat][gripper_qpos][gripper_qvel]
        # Size [7][7][7][7][2][2]
        data_dict['proprio'] = np.array(obs['robot0_proprio-state'], dtype=np.float32)
        for cam_name in self.cameras:
            data_dict[cam_name] = np.array(obs[cam_name + "_image"], dtype=np.uint8)
        data_dict['action'] = np.array(action, dtype=np.float32)
        data_dict['language_instruction'] = self.task

        self.episode = np.append(self.episode, data_dict)

    def save(self) -> None:
        cur_len = len(self.episode)
        if cur_len < 10:
            print(f'Only found {cur_len} Not enough steps to save episode')
            return
        if cur_len > self.episode_len:
            print(f'recording currently of size {cur_len} is longer than expected, skipping save')
            return

        # padding to episode_len        
        pad_len = self.episode_len - cur_len
        exemplar = self.episode[0]
        data_dict = {}
        data_dict['proprio'] = np.zeros_like(exemplar['proprio'])
        for cam_name in self.cameras:
            data_dict[cam_name] = np.zeros_like(exemplar[cam_name])
        data_dict['action'] = np.zeros_like(exemplar['action'])
        data_dict['language_instruction'] = self.task
        padding = np.array([data_dict] * pad_len)
        
        self.episode = np.append(self.episode, padding)

        idx = len([name for name in os.listdir(self.save_dir) if os.path.isfile(os.path.join(self.save_dir, name))])

        dataset_path = os.path.join(self.save_dir, f'episode_{idx}')
        
        np.save(dataset_path, self.episode)
