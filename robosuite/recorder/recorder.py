import h5py
import numpy as np
import os
import threading
import tensorflow as tf
import tensorflow_datasets as tfds


class Recorder:
    QPOS      = '/observations/qpos'
    QVEL      = '/observations/qvel'
    KEY_FRAME = '/observations/key_frame'
    ACTION    = '/action'

    @staticmethod
    def _cam_key(cam_name: str) -> str:
        return f'/observations/images/{cam_name}'

    def __init__(self, cameras, cam_height, cam_width, episode_len, task, data_dir, version) -> None:
        self.episodes_dir = f"{data_dir}/{task}/episodes/{version}"
        self._lock = threading.Lock()
        self.cameras = cameras
        self.task = task
        self.cam_height = cam_height
        self.cam_width = cam_width
        self.episode_len = episode_len
        self.data_dict = self._empty_data_dict()

    def _empty_data_dict(self) -> dict:
        d = {self.QPOS: [], self.QVEL: [], self.KEY_FRAME: [], self.ACTION: []}
        for cam_name in self.cameras:
            d[self._cam_key(cam_name)] = []
        return d

    @staticmethod
    def _discretize_grasp(action):
        a = action.copy()
        a[-1] = 1.0 if a[-1] >= 0.1 else -1.0
        return a

    def _record_obs(self, obs, key_frame):
        qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
        self.data_dict[self.QPOS].append(np.concatenate((qpos, obs['grasp'])))
        self.data_dict[self.QVEL].append(np.concatenate((obs['robot0_joint_vel'], obs['grasp'])))
        for cam_name in self.cameras:
            self.data_dict[self._cam_key(cam_name)].append(obs[cam_name + "_image"])
        self.data_dict[self.KEY_FRAME].append(key_frame)

    def record(self, obs, action, key_frame) -> None:
        with self._lock:
            self.data_dict[self.ACTION].append(self._discretize_grasp(action))
            self._record_obs(obs, key_frame)

    def reset(self) -> None:
        with self._lock:
            self.data_dict = self._empty_data_dict()

    def save(self) -> str:
        # Snapshot under the lock so record() can proceed immediately.
        with self._lock:
            snapshot = {k: list(v) for k, v in self.data_dict.items()}

        max_timesteps = len(snapshot[self.QPOS])
        if max_timesteps < 10:
            print('Not enough steps to save episode')
            return
        if max_timesteps > self.episode_len:
            print('recording longer than expected, skipping save')
            return

        # padding to episode_len
        pad_len = self.episode_len - max_timesteps
        snapshot[self.QPOS] = np.pad(snapshot[self.QPOS], ((0, pad_len), (0, 0)), mode='constant')
        snapshot[self.QVEL] = np.pad(snapshot[self.QVEL], ((0, pad_len), (0, 0)), mode='constant')
        action_full_pad = np.full((pad_len, len(snapshot[self.ACTION][0])), 0.0)
        snapshot[self.ACTION] = np.concatenate((snapshot[self.ACTION], action_full_pad))
        for cam_name in self.cameras:
            k = self._cam_key(cam_name)
            snapshot[k] = np.pad(snapshot[k], ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
        snapshot[self.KEY_FRAME] = np.pad(snapshot[self.KEY_FRAME], (0, pad_len), mode='constant')

        os.makedirs(self.episodes_dir, exist_ok=True)
        # Atomically claim a unique episode index by creating the file exclusively.
        idx = 0
        while True:
            dataset_path = os.path.join(self.episodes_dir, f'episode_{idx}')
            try:
                fd = os.open(dataset_path + '.hdf5', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                idx += 1

        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['real_len'] = max_timesteps
            obs_grp = root.create_group('observations')
            img_grp = obs_grp.create_group('images')
            for cam_name in self.cameras:
                _ = img_grp.create_dataset(cam_name, (self.episode_len, self.cam_height, self.cam_width, 3),
                                           dtype='uint8', chunks=(1, self.cam_height, self.cam_width, 3))
            _ = obs_grp.create_dataset('qpos', (self.episode_len, 8))
            _ = obs_grp.create_dataset('qvel', (self.episode_len, 8))
            _ = root.create_dataset('action', (self.episode_len, len(snapshot[self.ACTION][0])))
            _ = obs_grp.create_dataset('key_frame', (self.episode_len,), dtype='bool')

            for name, array in snapshot.items():
                root[name][...] = array
        return dataset_path + '.hdf5'

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
