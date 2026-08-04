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


# --- camera sets -------------------------------------------------------------
# The two models consume deliberately different views; keep them separate.
#
# The plain ACT policy is conditioned on the current frames alone, so it wants
# the wrist camera -- close-up geometry at the gripper is what tells it how to
# servo. sideview is NOT in this set: it adds a third full-scene backbone pass
# for a view whose value (a second, orthogonal measurement of the 3D path) only
# pays off when there are image-plane waypoints to disambiguate.
ACT_CAMERA_NAMES = ["robot0_eye_in_hand", "frontview", "birdview"]

# The waypoint-conditioned policy is conditioned on 2D waypoints, so it wants the
# arena's FIXED cameras: their static pose is what lets generate_waypoints.py
# project the recorded 3D eef path into each image plane offline. Their optical
# axes are mutually near-orthogonal (frontview looks along -x, birdview along -z,
# sideview along -y), so every world axis is measured in-plane by two of the
# three views. The wrist camera moves every step and so cannot be projected
# into -- it is deliberately absent.
WAYPOINT_CAMERA_NAMES = ["frontview", "birdview", "sideview"]

# Cameras written into every recording: the union, so one recording trains either
# model. Each model then selects its own subset by name, so the extra views cost
# render time and disk but never reach a model that did not ask for them.
RECORDING_CAMERA_NAMES = list(dict.fromkeys(ACT_CAMERA_NAMES + WAYPOINT_CAMERA_NAMES))


# --- waypoint episode length -------------------------------------------------
# The waypoint policy predicts one action chunk covering the whole episode from a
# single conditioning frame, so its chunk length (num_queries) IS the supported
# episode length. Recordings are shorter or longer than this in practice, so
# WaypointDataset pads/truncates every example to exactly this many steps -- the
# recording length no longer has to match the model. Raising it lets longer
# episodes train end-to-end; it also grows decoder self-attention quadratically,
# so expect to lower batch_size_train alongside it.
WAYPOINT_EPISODE_LEN = 600


# task config (you can add new tasks)
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 600,
    'state_dim': 8,
    'action_dim': 7,
    'cam_width': 256,
    'cam_height': 256,
    # ACT training reads camera_names from here while ACT inference reads it from
    # POLICY_CONFIG, so both must name the same views -- share one constant.
    'camera_names': ACT_CAMERA_NAMES,
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
    'camera_names': ACT_CAMERA_NAMES,
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
    'camera_names': ACT_CAMERA_NAMES,
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

# waypoint-conditioned policy config (parallel to POLICY_CONFIG).
# `camera_names` must match the cameras passed to generate_waypoints.py -- the
# waypoint JSONs only contain those cameras. `max_waypoints` is filled in at
# runtime from the waypoint dataset's manifest.json.
WAYPOINT_POLICY_CONFIG = {
    'lr': 1e-4,
    'device': device,
    # One chunk spans the whole episode, so this must equal WAYPOINT_EPISODE_LEN.
    'num_queries': WAYPOINT_EPISODE_LEN,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': WAYPOINT_CAMERA_NAMES,
    'policy_class': 'ACT',
    'temporal_agg': False,
    'state_dim': 8,
    'action_dim': 7,
    'max_waypoints': None,  # set from manifest.json at train time
    'waypoint_dim': 3,      # [y, x, grip]
}

# waypoint training config
WAYPOINT_TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 1000,
    # Decoder self-attention is O(num_queries^2 * batch), so a 1000-step chunk
    # costs ~1.6 GiB of activations per sample: batch 8 OOMs on an 8 GB card,
    # batch 4 peaks at ~6.6 GiB. Raise this on a larger GPU.
    'batch_size_val': 8,
    'batch_size_train': 4,
    'eval_ckpt_name': 'policy_last.ckpt',
    # Steps every example is padded/truncated to; must match num_queries so the
    # predicted chunk and the target sequence line up in the L1 loss.
    'episode_len': WAYPOINT_EPISODE_LEN,
}