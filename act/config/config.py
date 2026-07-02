import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'robosuite/data/'

device = ''
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    raise RuntimeError('No GPU found (CUDA or MPS required)')
os.environ['DEVICE'] = device


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 600,
    'state_dim': 8,
    'action_dim': 7,
    'cam_width': 256,
    'cam_height': 256,
    'camera_names': ["robot0_eye_in_hand", "frontview", "birdview"],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ["robot0_eye_in_hand", "frontview", "birdview"],
    'policy_class': 'ACT',
    'temporal_agg': False,
    'state_dim': 8,
    'action_dim': 7
}

# finetuning config
FINETUNING_POLICY_CONFIG = {
    'lr': 1e-6,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-7,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ["robot0_eye_in_hand", "frontview", "birdview"],
    'policy_class': 'ACT',
    'temporal_agg': False,
    'state_dim': 8,
    'action_dim': 7
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 1000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
}

# finetuning_train_config
FINETUNING_TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 1000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
}