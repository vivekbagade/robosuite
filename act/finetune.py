import tempfile
from config.config import FINETUNING_POLICY_CONFIG, TASK_CONFIG, FINETUNING_TRAIN_CONFIG, TRAIN_CONFIG  # must import first

import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import shutil

from utils.utils import *
from dataset.episodicdataset import load_data, get_norm_stats, get_combined_norm_stats
from absl import flags
from model_tree import ModelTree
from train import ACTTrainer


BASE_VERSION = flags.DEFINE_string(
    "base-version", "1.0.0", "version to finetune"
)

NEW_VERSION = flags.DEFINE_string(
    "new-version", "1.1.0", "version to create after finetuning"
)

TASK = flags.DEFINE_string(
    "task", "PickPlaceCan", "The task to automate"
)

DATA_DIR = flags.DEFINE_string(
    "data_dir", "/act-data", "The dir containing training episodes, weights etc"
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)
args = FLAGS

base_weights_dir = get_weights_dir(DATA_DIR.value, TASK.value, BASE_VERSION.value)
checkpoint_dir = get_weights_dir(DATA_DIR.value, TASK.value, NEW_VERSION.value)
new_episodes_dir = get_episodes_dir(DATA_DIR.value, TASK.value, NEW_VERSION.value)


# configs
task_cfg = TASK_CONFIG
train_cfg = FINETUNING_TRAIN_CONFIG
policy_config = FINETUNING_POLICY_CONFIG


def copy_base_episodes(new_dir, base_versions):
    num_new_episodes = count_episodes(new_dir)
    idx = count_episodes(new_dir)
    num_episodes_to_copy = 3 * num_new_episodes
    src_list = [get_episodes_dir(DATA_DIR.value, TASK.value, version) for version in base_versions]
    files_to_copy = get_file_paths_to_copy(src_list, num_episodes_to_copy)
    for file in files_to_copy:
        print(f"Copying {file} to {new_dir}")
        shutil.copy(file, os.path.join(new_dir, f'episode_{idx}.hdf5'))
        idx += 1

# Copy episodes to /tmp/act-finetuning-xxx and return the path
def copy_to_tmp(base_dir):
    # create a temporary directory
    dir = tempfile.mkdtemp(prefix='act-finetuning-')
    print(f'Copying episodes to {dir}')
    # copy all files from base_dir to dir
    for file in os.listdir(base_dir):
        if file.endswith('.hdf5'):
            shutil.copy(os.path.join(base_dir, file), os.path.join(dir, file))
    return dir
    

if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Copy episodes to /tmp/act-finetuning-xxx
    tmp_episodes_dir = copy_to_tmp(new_episodes_dir)
    num_new_episodes = count_episodes(tmp_episodes_dir)
    
    model_tree = ModelTree.load_from_disk(DATA_DIR.value, TASK.value)
    model_tree.add_new_version(NEW_VERSION.value, {'description': 'Finetuning test'})
    model_tree.add_weight_edge(base_version=BASE_VERSION.value, new_version=NEW_VERSION.value)
    base_episode_edges = model_tree.get_episode_edges(BASE_VERSION.value)
    base_episode_edges.append(BASE_VERSION.value)  # include the base version itself
    for edge in base_episode_edges:
        model_tree.add_episode_edge(base_version=edge, new_version=NEW_VERSION.value)


    # To stop catastrophic forgetting
    copy_base_episodes(tmp_episodes_dir, base_episode_edges)
    num_episodes = count_episodes(tmp_episodes_dir)

    # get norm stats
    # load base stats
    base_stats_path = os.path.join(base_weights_dir, 'dataset_stats.pkl')
    with open(base_stats_path, 'rb') as f:
        base_stats = pickle.load(f)
    num_base = base_stats.get('n')
    if num_base is None:
        print("Warning: base stats missing 'n' (old format). Combined stats may be inaccurate.")
        num_base = 1

    # calculate stats for new episodes only (before base episodes were appended to tmp dir)
    new_stats = get_norm_stats(tmp_episodes_dir, num_new_episodes)
    num_new = new_stats['n']

    # combine stats
    stats = get_combined_norm_stats(base_stats, new_stats, num_base, num_new)
    print(f'Combined stats: {stats}')

    # load data — pass combined stats so training and inference use identical normalization
    train_dataloader, val_dataloader, _, _ = load_data(tmp_episodes_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'],
                                                            norm_stats=stats)
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    trainer = ACTTrainer(policy_config, train_cfg, base_weights_dir, checkpoint_dir)
    trainer.train_bc(train_dataloader, val_dataloader)

    # delete the tmp dir
    shutil.rmtree(tmp_episodes_dir)

    model_tree.save_to_disk()
