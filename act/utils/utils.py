from collections import defaultdict
import os
import random
import numpy as np
import torch
from einops import rearrange

def get_image(obs, camera_names, device='cpu'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs[cam_name + "_image"], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_file_paths_to_copy(src_list, num_to_copy):
    """
    Determines a list of files to copy from source directories, ensuring the
    number from each source is as equal as possible.

    Args:
        src_list (list): A list of source directory paths.
        num_to_copy (int): The total number of files to select.

    Returns:
        list: A list of absolute file paths for the selected files.
    """
    if num_to_copy == 0:
        return []

    # 1. Collect all available files from each source directory
    available_files = defaultdict(list)
    total_available = 0

    for src_dir in src_list:
        # List hdf5 only files, not sub-directories
        files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f)) and f.endswith('.hdf5')]
        if len(files) == 0:
            continue
        available_files[src_dir] = files
        total_available += len(files)

    # Adjust num_to_copy if not enough files are available
    if total_available < num_to_copy:
        raise ValueError(f"Not enough files available to copy. Requested: {num_to_copy}, Available: {total_available}")

    # 2. Create a "selection plan" using a round-robin method
    selection_plan = defaultdict(int)
    plan_count = 0
    active_dirs = [d for d in src_list if d in available_files]

    if not active_dirs:
        raise ValueError("No directories with available files.")

    while plan_count < num_to_copy:
        files_added_this_round = 0
        for src_dir in active_dirs:
            # If we still need files and this directory has more to offer...
            if plan_count < num_to_copy and selection_plan[src_dir] < len(available_files[src_dir]):
                selection_plan[src_dir] += 1
                plan_count += 1
                files_added_this_round += 1
        
        # If a full pass adds no files, all sources are exhausted
        if files_added_this_round == 0:
            break

    # 3. Build the final list of absolute paths based on the plan
    file_paths_to_copy = []
    for src_dir, num_to_take in selection_plan.items():
        # Get the specific filenames to take from this directory
        # Randomly select files
        files_to_select = random.sample(available_files[src_dir], num_to_take)
        for filename in files_to_select:
            absolute_path = os.path.join(src_dir, filename)
            file_paths_to_copy.append(absolute_path)
            
    return file_paths_to_copy

def get_episodes_dir(data_dir: str, task: str, version: str):
    return f"{data_dir}/{task}/episodes/{version}"

def get_weights_dir(data_dir: str, task: str, version: str):
    return f"{data_dir}/{task}/weights/{version}"

def get_eval_result_file(data_dir: str, task: str, version: str):
    return os.path.join(get_episodes_dir(data_dir, task, version), f"eval-result-{task}-{version}-sim.txt")
