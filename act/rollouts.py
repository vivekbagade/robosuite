"""Use ACT policy to eval can pick and place.

"""
import pickle
from config.config import POLICY_CONFIG, TRAIN_CONFIG, device # must import first
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.recorder import Recorder
from policy import ACTPolicy
import torch
import os
import absl.flags as flags
import sys

from utils.utils import get_image

collision_init_time = 100

if __name__ == "__main__":

    flags.DEFINE_string("environment", "Lift", "Environment to use")
    flags.DEFINE_list("robots", ["Panda"], "Which robot(s) to use in the env")
    flags.DEFINE_string("config", "single-arm-opposed", "Specified environment configuration if necessary")
    flags.DEFINE_string("arm", "right", "Which arm to control (eg bimanual) 'right' or 'left'")
    flags.DEFINE_boolean("switch_on_grasp", False, "Switch gripper control on gripper action")
    flags.DEFINE_boolean("toggle_camera_on_grasp", False, "Switch camera angle on gripper action")
    flags.DEFINE_string("controller", "osc", "Choice of controller. Can be 'ik' or 'osc'")
    flags.DEFINE_string("device", "keyboard", "Device to use for control")
    flags.DEFINE_float("pos_sensitivity", 1.0, "How much to scale position user inputs")
    flags.DEFINE_float("rot_sensitivity", 1.0, "How much to scale rotation user inputs")
    flags.DEFINE_string("data_dir", "/act-data", "The dir containing training episodes, weights etc")
    flags.DEFINE_string("version", "1.0.0", "The version of the model to eval")
    flags.DEFINE_integer("num_episodes", 1, "Number of episodes to run for evaluation")
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    # Parse command line arguments
    args = FLAGS
    checkpoint_dir = f"{args.data_dir}/{args.environment}/weights/{args.version}"

    # Import controller config for EE IK or OSC (pos/ori)
    if args.controller == "ik":
        controller_name = "IK_POSE"
    elif args.controller == "osc":
        controller_name = "OSC_POSE"
    else:
        print("Error: Unsupported controller specified. Must be either 'ik' or 'osc'!")
        raise ValueError

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Check if we're using a multi-armed environment and use env_configuration argument if so
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None
    
    config["obj_pos_override"] = [0.210, -0.407, 0.885]

    # Create environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=["robot0_eye_in_hand", "frontview", "birdview"],
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    ckpt_path = os.path.join(checkpoint_dir, TRAIN_CONFIG['eval_ckpt_name'])
    policy = ACTPolicy(POLICY_CONFIG)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    camera_names = POLICY_CONFIG['camera_names']
    query_frequency = POLICY_CONFIG['num_queries']
    current_ncon = 0

    for i in range(args.num_episodes):
        obs = env.reset()
        all_actions = None
        print(f"Episode {i+1} in progress...")
        recorder = Recorder(["robot0_eye_in_hand", "frontview", "birdview"],
                         256, 256, 800, "PickPlaceCan", "/act-data", f"{args.version}-sim")
        for t in range(800):
            qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
            grasp = [0]
            if 'grasp' in obs:
                grasp = [obs['grasp']]
            qpos = np.concatenate((qpos, grasp))
            qpos = pre_process(qpos)
            qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

            

            with torch.inference_mode():
                if t % query_frequency == 0:
                    all_actions = policy(qpos, get_image(obs, camera_names, device))

                cur_action = all_actions[:, t % query_frequency]
                cur_action = cur_action.squeeze(0).cpu().numpy()
                cur_action = post_process(cur_action)

            # If action is none, then this a reset so we should break
            if cur_action is None:
                print('No action')
                break
            # record the current obs and corresponding action picked
            obs['grasp'] = np.array([0]) if grasp == -1 else np.array([1])
            key_frame = False
            # Check if the number of contacts has changed, if so, record a key frame
            if abs(current_ncon - env.sim.data.ncon) > 0 and i >= collision_init_time:
                key_frame = True
                current_ncon = env.sim.data.ncon
            recorder.record(obs, cur_action, key_frame)

            obs, reward, done, info = env.step(cur_action)
            
            
            env.render()
        # Save the episode data
        recorder.save()
    print("End of episode")
