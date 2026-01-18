import tempfile
from critic_record import CriticRecord
from config.config import FINETUNING_POLICY_CONFIG, TASK_CONFIG, FINETUNING_TRAIN_CONFIG  # must import first

import os
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import shutil

from utils.utils import *
from dataset.episodicdataset import load_data
from policy import ACTPolicy
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

TASK_DEFINITION = flags.DEFINE_string(
    "task_definition", "The robot should pick up the red can and place it in the right bin. The right bin has a silhouette of a can on it.",
    "The task definition to use for evaluation"
)

FLAGS = flags.FLAGS
FLAGS(sys.argv)

base_weights_dir = get_weights_dir(DATA_DIR.value, TASK.value, BASE_VERSION.value)
checkpoint_dir = get_weights_dir(DATA_DIR.value, TASK.value, NEW_VERSION.value)
new_episodes_dir = get_episodes_dir(DATA_DIR.value, TASK.value, NEW_VERSION.value)


# configs
task_cfg = TASK_CONFIG
train_cfg = FINETUNING_TRAIN_CONFIG
policy_config = FINETUNING_POLICY_CONFIG


def copy_successful_base_episodes(base_version):
    critic_record = CriticRecord(DATA_DIR.value, TASK.value, f'{base_version}-sim', TASK_DEFINITION.value)
    episode_paths = critic_record.read_successful_episodes()
    if len(episode_paths) == 0:
        print(f"No successful episodes found for version {base_version}. Please check the directory.")
        return None
    new_dir = tempfile.mkdtemp(prefix='act-finetuning-')
    print(f'Copying {len(episode_paths)} base episodes to {new_dir}')
    # copy all files from dir to new_dir
    idx = 0
    for ep in episode_paths:
        print("Copying episode:", ep)
        shutil.copy(ep, os.path.join(new_dir, f'episode_{idx}.hdf5'))
        idx += 1
    return new_dir

if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_tree = ModelTree.load_from_disk(DATA_DIR.value, TASK.value)
    model_tree.add_new_version(NEW_VERSION.value, {'description': f'Reinforced version of {BASE_VERSION.value}'})
    model_tree.add_weight_edge(base_version=BASE_VERSION.value, new_version=NEW_VERSION.value)
    model_tree.add_episode_edge(base_version=BASE_VERSION.value, new_version=NEW_VERSION.value)

    # Copy episodes to /tmp/act-finetuning-xxx
    tmp_base_episodes_dir = copy_successful_base_episodes(BASE_VERSION.value)
    if tmp_base_episodes_dir is None:
        sys.exit(1)
    num_episodes = len(os.listdir(tmp_base_episodes_dir))

    print(f'Number of episodes to train on: {num_episodes}') # 5 46 44
    print(f'Episodes dir: {tmp_base_episodes_dir}')

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(tmp_base_episodes_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    trainer = ACTTrainer(policy_config, train_cfg, base_weights_dir, checkpoint_dir)

    # train
    trainer.train_bc(train_dataloader, val_dataloader)

    # delete the tmp dir
    shutil.rmtree(tmp_base_episodes_dir)

    model_tree.save_to_disk()

