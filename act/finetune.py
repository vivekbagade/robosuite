import tempfile
from config.config import FINETUNING_POLICY_CONFIG, TASK_CONFIG, FINETUNING_TRAIN_CONFIG, TRAIN_CONFIG, POLICY_CONFIG  # must import first

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

base_weights_dir = f"{DATA_DIR.value}/{TASK.value}/weights/{BASE_VERSION.value}"
checkpoint_dir = f"{DATA_DIR.value}/{TASK.value}/weights/{NEW_VERSION.value}"
new_episodes_dir = f"{DATA_DIR.value}/{TASK.value}/episodes/{NEW_VERSION.value}"
base_episodes_dir = f"{DATA_DIR.value}/{TASK.value}/episodes/{BASE_VERSION.value}"


# configs
task_cfg = TASK_CONFIG
train_cfg = FINETUNING_TRAIN_CONFIG
policy_config = FINETUNING_POLICY_CONFIG

# device
device = os.environ['DEVICE']

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def train_bc(train_dataloader, val_dataloader, policy_config):
    # load policy
    ckpt_path = os.path.join(base_weights_dir, TRAIN_CONFIG['eval_ckpt_name'])
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(train_cfg['num_epochs']):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 200 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)


def copy_base_episodes(base_dir, new_dir):
    base_episodes_names = [name for name in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, name))]
    num_base_episodes = len(base_episodes_names)
    num_new_episodes = len([name for name in os.listdir(new_dir) if os.path.isfile(os.path.join(new_dir, name))])
    
    if 2 * num_new_episodes > num_base_episodes or num_new_episodes == 0:
        raise IndexError('Num of episodes in base should be at least 2x more than ones in new dir')

    # Pick the same number of episodes from the base dir to the new dir
    random_files = random.sample(base_episodes_names, 2*num_new_episodes)
    idx = num_new_episodes
    for file in random_files:
        shutil.copy(os.path.join(base_dir, file), os.path.join(new_dir, f'episode_{idx}.hdf5'))
        idx+=1

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
    # number of training episodes
    num_episodes = len(os.listdir(new_episodes_dir))

    # Copy episodes to /tmp/act-finetuning-xxx
    new_episodes_dir = copy_to_tmp(new_episodes_dir)

    # To stop catastrophic forgetting
    copy_base_episodes(base_episodes_dir, new_episodes_dir)

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(new_episodes_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_bc(train_dataloader, val_dataloader, policy_config)

    # delete the tmp dir
    shutil.rmtree(new_episodes_dir)