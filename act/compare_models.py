import os
from config.config import TRAIN_CONFIG, POLICY_CONFIG
from absl import flags
import sys
import torch
from policy import ACTPolicy

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

train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
device = os.environ['DEVICE']


def print_model_comparison(policy1, policy2):
    params1 = torch.cat([p.view(-1) for p in policy1.parameters()])
    params2 = torch.cat([p.view(-1) for p in policy2.parameters()])

    l2_distance = torch.linalg.norm(params1 - params2, ord=2)
    print(f"L2 Distance between model weights: {l2_distance.item():.4f}")

    # L1 Norm (Manhattan Distance)
    l1_distance = torch.linalg.norm(params1 - params2, ord=1)
    print(f"L1 Distance between model weights: {l1_distance.item():.4f}")

    # Cosine Similarity
    # Note: Use torch.nn.functional for this
    cos_sim = torch.nn.functional.cosine_similarity(params1, params2, dim=0)
    print(f"Cosine Similarity between model weights: {cos_sim.item():.4f}")

if __name__ == "__main__":
    # Load the base policy
    base_weights_dir = f"{DATA_DIR.value}/{TASK.value}/weights/{BASE_VERSION.value}"
    base_policy_path = os.path.join(base_weights_dir, "policy_last.ckpt")
    if not os.path.exists(base_policy_path):
        raise FileNotFoundError(f"Base policy checkpoint {base_policy_path} does not exist.")
    
    ckpt_path = os.path.join(base_weights_dir, TRAIN_CONFIG['eval_ckpt_name'])
    print(f"Checkpoint path: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist. Please check the path.")
    base_policy = ACTPolicy(policy_config)
    loading_status = base_policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(f"Loading status: {loading_status}")

    # Load the new policy
    new_weights_dir = f"{DATA_DIR.value}/{TASK.value}/weights/{NEW_VERSION.value}"
    new_policy_path = os.path.join(new_weights_dir, "policy_last.ckpt")
    if not os.path.exists(new_policy_path):
        raise FileNotFoundError(f"New policy checkpoint {new_policy_path} does not exist.")
    new_policy = ACTPolicy(policy_config)
    loading_status = new_policy.load_state_dict(torch.load(new_policy_path, map_location=torch.device(device)))
    print(f"Loading status: {loading_status}")

    # Compare the two policies
    print_model_comparison(base_policy, new_policy)

