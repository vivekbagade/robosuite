import h5py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        #self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            real_len = int(root.attrs.get('real_len', episode_len))
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(real_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1
        # mask recorder-level padding (zero images, zero motion) — not covered by dataset-level is_pad
        offset = start_ts if is_sim else max(0, start_ts - 1)
        recorder_pad_start = real_len - offset
        if recorder_pad_start < action_len:
            is_pad[max(0, recorder_pad_start):action_len] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    example_qpos = None
    total_real_steps = 0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
            real_len = int(root.attrs.get('real_len', len(action)))
        all_qpos_data.append(torch.from_numpy(qpos[:real_len]))
        all_action_data.append(torch.from_numpy(action[:real_len]))
        total_real_steps += real_len
        example_qpos = qpos
    # cat across episodes so padding frames are excluded from statistics
    all_qpos_data = torch.cat(all_qpos_data)    # (total_real_steps, state_dim)
    all_action_data = torch.cat(all_action_data) # (total_real_steps, action_dim)

    # normalize action data
    action_mean = all_action_data.mean(dim=0, keepdim=True)
    action_std = all_action_data.std(dim=0, keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": example_qpos, "n": total_real_steps}

    return stats

def combined_sdev_mean(n1, mean1, std1, n2, mean2, std2):
    combined_mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    total = n1 + n2

    numerator = (
        (n1 - 1) * std1**2
        + (n2 - 1) * std2**2
        + n1 * n2 * (mean1 - mean2)**2 / total
    )
    combined_std = np.sqrt(numerator / (total - 1))

    return combined_std, combined_mean

def get_combined_norm_stats(base_stats, new_stats, num_base, num_new):
    comb_qpos_std, comb_qpos_mean = combined_sdev_mean(num_base, base_stats['qpos_mean'], base_stats['qpos_std'],
                                    num_new, new_stats['qpos_mean'], new_stats['qpos_std'])
    comb_action_std, comb_action_mean = combined_sdev_mean(num_base, base_stats['action_mean'], base_stats['action_std'],
                                      num_new, new_stats['action_mean'], new_stats['action_std'])
    combined_stats = {
        "action_mean": comb_action_mean,
        "action_std": comb_action_std,
        "qpos_mean": comb_qpos_mean,
        "qpos_std": comb_qpos_std,
        "example_qpos": new_stats['example_qpos'],
        "n": num_base + num_new
    }
    return combined_stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, norm_stats=None):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    if norm_stats is None:
        norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim
