import tempfile
from config.config import FINETUNING_POLICY_CONFIG, TASK_CONFIG, FINETUNING_TRAIN_CONFIG  # must import first

import os
import pickle
import sys
import shutil

from utils.utils import *
from dataset.episodicdataset import load_data, get_norm_stats, get_combined_norm_stats
from absl import flags
from model_tree import ModelTree
from train import ACTTrainer
from critic_record import CriticRecord


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
    "data-dir", "/act-data", "The dir containing training episodes, weights etc"
)

FROM_SIM = flags.DEFINE_boolean(
    "from-sim", False,
    "Finetune on the critic-approved rollouts of {base_version}-sim (reinforce) instead "
    "of the episodes recorded under new-version"
)
    
TASK_DEFINITION = flags.DEFINE_string(
    "task-definition", "The robot should pick up the red can and place it in the right bin. The right bin has a silhouette of a can on it.",
    "The task definition; used to locate the critic record when --from_sim"
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

# anti-forgetting mix: use BASE_TO_NEW_RATIO base episodes per new episode
BASE_TO_NEW_RATIO = 3


def get_new_episode_paths():
    """Absolute paths to the new demonstrations to finetune on.

    With --from_sim, reinforce on the model's own rollouts: the critic-approved
    episodes of {base_version}-sim. Otherwise use the episodes recorded under
    new-version.
    """
    if FROM_SIM.value:
        critic_record = CriticRecord(DATA_DIR.value, TASK.value, f'{BASE_VERSION.value}-sim', TASK_DEFINITION.value)
        paths = critic_record.read_successful_episodes()
        if len(paths) == 0:
            raise ValueError(f"No successful episodes found for {BASE_VERSION.value}-sim; nothing to finetune on.")
        return [os.path.abspath(p) for p in paths]

    if not os.path.isdir(new_episodes_dir) or count_episodes(new_episodes_dir) == 0:
        raise ValueError(f"No new episodes found in {new_episodes_dir}; nothing to finetune on.")
    hdf5_files = sorted(f for f in os.listdir(new_episodes_dir) if f.endswith('.hdf5'))
    return [os.path.abspath(os.path.join(new_episodes_dir, f)) for f in hdf5_files]


def copy_base_episodes(new_dir, base_versions, num_to_copy):
    idx = count_episodes(new_dir)
    src_list = [get_episodes_dir(DATA_DIR.value, TASK.value, version) for version in base_versions]
    files_to_copy = get_file_paths_to_copy(src_list, num_to_copy)
    for file in files_to_copy:
        print(f"Linking {file} to {new_dir}")
        os.symlink(os.path.abspath(file), os.path.join(new_dir, f'episode_{idx}.hdf5'))
        idx += 1

# Symlink the given episodes into /tmp/act-finetuning-xxx and return the path
def link_to_tmp(episode_paths):
    # create a temporary directory
    dir = tempfile.mkdtemp(prefix='act-finetuning-')
    print(f'Linking {len(episode_paths)} episodes into {dir}')
    # Re-index files contiguously as episode_0..episode_{n-1} regardless of the
    # source naming. get_norm_stats and EpisodicDataset iterate range(num_episodes)
    # assuming contiguous episode_{idx}.hdf5 names, and copy_base_episodes appends
    # at idx = count_episodes, so the new episodes must occupy indices 0..n-1.
    # Episodes are read-only during finetuning, so symlinks avoid duplicating
    # potentially large HDF5 files; rmtree on the tmp dir removes only the links.
    for idx, path in enumerate(episode_paths):
        os.symlink(os.path.abspath(path), os.path.join(dir, f'episode_{idx}.hdf5'))
    return dir


if __name__ == '__main__':
    # set seed
    set_seed(train_cfg['seed'])

    # resolve the source of new demonstrations (fails fast if there are none)
    new_episode_paths = get_new_episode_paths()

    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_tree = ModelTree.load_from_disk(DATA_DIR.value, TASK.value)
    description = f'Reinforced version of {BASE_VERSION.value}' if FROM_SIM.value else 'Finetuning test'
    model_tree.add_new_version(NEW_VERSION.value, {'description': description})
    model_tree.add_weight_edge(base_version=BASE_VERSION.value, new_version=NEW_VERSION.value)
    base_episode_edges = model_tree.get_episode_edges(BASE_VERSION.value)
    base_episode_edges.append(BASE_VERSION.value)  # include the base version itself
    for edge in base_episode_edges:
        model_tree.add_episode_edge(base_version=edge, new_version=NEW_VERSION.value)

    # Decide up front exactly how many new and base episodes to use, keeping the
    # base:new ratio at BASE_TO_NEW_RATIO:1 for anti-forgetting. If the base lineage
    # doesn't have enough episodes to match, use proportionally fewer new examples.
    base_src_list = [get_episodes_dir(DATA_DIR.value, TASK.value, v) for v in base_episode_edges]
    num_base_available = sum(count_episodes(d) for d in base_src_list if os.path.isdir(d))
    num_new = min(len(new_episode_paths), num_base_available // BASE_TO_NEW_RATIO)
    if num_new == 0:
        raise ValueError(
            f"Not enough base episodes to finetune: {num_base_available} available across "
            f"{base_episode_edges}, need at least {BASE_TO_NEW_RATIO} per new episode."
        )
    num_base = BASE_TO_NEW_RATIO * num_new
    new_episode_paths = new_episode_paths[:num_new]
    print(f'Using {num_new} new episodes and {num_base} base episodes (base:new = {BASE_TO_NEW_RATIO}:1)')

    # Symlink episodes into /tmp/act-finetuning-xxx
    tmp_episodes_dir = link_to_tmp(new_episode_paths)
    try:
        num_new_episodes = count_episodes(tmp_episodes_dir)

        # To stop catastrophic forgetting
        copy_base_episodes(tmp_episodes_dir, base_episode_edges, num_base)
        num_episodes = count_episodes(tmp_episodes_dir)

        # get norm stats
        # load base stats
        base_stats_path = os.path.join(base_weights_dir, 'dataset_stats.pkl')
        with open(base_stats_path, 'rb') as f:
            base_stats = pickle.load(f)
        num_base = base_stats.get('n')
        if num_base is None:
            raise ValueError(f"Base stats file {base_stats_path} does not contain 'n' key")

        # calculate stats for new episodes only; link_to_tmp re-indexes the new
        # episodes to indices 0..num_new_episodes-1, so they precede the appended base episodes
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
        stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        # train
        trainer = ACTTrainer(policy_config, train_cfg, base_weights_dir, checkpoint_dir)
        trainer.train_bc(train_dataloader, val_dataloader)

        model_tree.save_to_disk()
    finally:
        # always clean up the tmp dir, even if finetuning fails partway through
        shutil.rmtree(tmp_episodes_dir, ignore_errors=True)
