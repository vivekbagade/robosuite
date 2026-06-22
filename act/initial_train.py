from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG # must import first

import os
import pickle
from absl import flags
import sys

from utils.utils import *
from dataset.episodicdataset import load_data
from train import ACTTrainer
from model_tree import ModelTree

VERSION = flags.DEFINE_string(
    "version", "1.0.0", "version to initially train"
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

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = base_weights_dir = get_weights_dir(DATA_DIR.value, TASK.value, VERSION.value)


if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    # number of training episodes
    data_dir = DATA_DIR.value
    
    episodes_dir = get_episodes_dir(DATA_DIR.value, TASK.value, VERSION.value)
    num_episodes = count_episodes(episodes_dir)

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(episodes_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    
    model_tree = ModelTree.load_from_disk(DATA_DIR.value, TASK.value)
    if model_tree is None:
        model_tree = ModelTree(DATA_DIR.value, TASK.value)
    model_tree.add_new_version(VERSION.value, {'description': 'Initial training'})

    # train
    trainer = ACTTrainer(policy_config, train_cfg, None, checkpoint_dir)
    trainer.train_bc(train_dataloader, val_dataloader)
    
    model_tree.save_to_disk()