"""Use ACT policy to eval can pick and place.

"""
import pickle
from config.config import POLICY_CONFIG, TRAIN_CONFIG, device # must import first
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
from robosuite.recorder import Recorder, grasp_state_from_command
from policy import ACTPolicy
import torch
import os
import absl.flags as flags
import sys

from utils.utils import *
from critic import Critic
from critic_record import CriticRecord

collision_init_time = 100


def print_progress_bar(results):
    """Print a persistent progress bar of rollout outcomes.

    Each rollout is a block: green for success, red for failure. Followed by
    num_successes/num_rollouts and the success percentage.
    """
    GREEN_BG = "\033[42m"
    RED_BG = "\033[41m"
    RESET = "\033[0m"

    num_rollouts = len(results)
    num_successes = sum(1 for r in results if r)
    percentage = (num_successes / num_rollouts * 100) if num_rollouts else 0

    bar = "".join(f"{GREEN_BG if r else RED_BG} {RESET}" for r in results)
    # \r returns to start of the line and \033[K clears it so the bar updates in place
    print(f"\r\033[KRollouts: [{bar}] {num_successes}/{num_rollouts} ({percentage:.1f}%)", end="", flush=True)


def get_query_frequency(args, num_queries):
    """Resolve how often to re-query the policy.

    An explicit override takes precedence over everything else. Otherwise, with
    temporal ensembling we re-query every step and blend overlapping chunks;
    without it we run open-loop, consuming one full chunk before re-querying.
    """
    if args.query_frequency is not None:
        return args.query_frequency
    return 1 if args.temporal_agg else num_queries


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
    flags.DEFINE_boolean("save", True, "Whether to save the episode or not")
    flags.DEFINE_string("task_definition", "The robot should pick up the red can and place it in the right bin. The right bin has a silhouette of a can on it.", "The task definition to use for evaluation")
    flags.DEFINE_boolean("critic", True, "Whether to use the critic to evaluate the episode or not")
    flags.DEFINE_boolean("temporal_agg", False, "Whether to use temporal ensembling (query every step and average overlapping predictions)")
    flags.DEFINE_float("temporal_agg_k", 0.1, "Exponential decay for temporal ensembling weights (smaller = smoother, larger = more reactive)")
    flags.DEFINE_integer("query_frequency", None, "Override for how often to re-query the policy. Takes precedence over temporal_agg-derived defaults")
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    # Parse command line arguments
    args = FLAGS
    checkpoint_dir = get_weights_dir(args.data_dir, args.environment, args.version)

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
    
    # config["obj_pos_override"] = [0.210, -0.407, 0.885]

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
    num_queries = POLICY_CONFIG['num_queries']
    max_timesteps = 800
    action_dim = POLICY_CONFIG['action_dim']
    query_frequency = get_query_frequency(args, num_queries)
    critic = Critic()
    critic_record = CriticRecord(args.data_dir, args.environment, f'{args.version}-sim', args.task_definition)

    rollout_results = []
    for i in range(args.num_episodes):
        current_ncon = 0
        obs = env.reset()
        # Latched commanded grasp state (qpos/qvel index 7): 0=open, 1=closed.
        # The gripper starts open and holds its last command until re-commanded,
        # so we feed the previous step's command as proprioception and update it
        # from the gripper command the policy chooses this step.
        grasp_state = np.array([0])
        all_actions = None
        # all_time_actions[i, j] = action that the query made at time i proposes for absolute time j
        all_time_actions = None
        if args.temporal_agg:
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, action_dim]
            ).to(device)
        recorder = Recorder(["robot0_eye_in_hand", "frontview", "birdview"],
                         256, 256, 800, args.environment, args.data_dir, f"{args.version}-sim")
        for t in range(max_timesteps):
            qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
            # Feed the latched grasp command (from the previous step) as proprioception.
            qpos = np.concatenate((qpos, grasp_state))
            qpos = pre_process(qpos)
            qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)

            

            with torch.inference_mode():
                if t % query_frequency == 0:
                    all_actions = policy(qpos, get_image(obs, camera_names, device))

                if args.temporal_agg:
                    # store this chunk against the absolute times it predicts: t .. t+num_queries-1
                    all_time_actions[[t], t:t + num_queries] = all_actions
                    # gather every past prediction that has an opinion about time t
                    actions_for_curr_step = all_time_actions[:, t]
                    populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[populated]
                    # rows are ordered oldest -> newest prediction; weight[0] (oldest) is highest.
                    # smaller temporal_agg_k => more uniform weights => faster incorporation of new obs.
                    exp_weights = np.exp(-args.temporal_agg_k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                cur_action = raw_action.squeeze(0).cpu().numpy()
                cur_action = post_process(cur_action)

            # Update the latched grasp state from the gripper command just chosen,
            # then record it so qpos[7] matches the recorded (discretized) action command.
            grasp_state = grasp_state_from_command(cur_action[-1])
            obs['grasp'] = grasp_state
            key_frame = False
            # Check if the number of contacts has changed, if so, record a key frame
            if abs(current_ncon - env.sim.data.ncon) > 0 and t >= collision_init_time:
                key_frame = True
            current_ncon = env.sim.data.ncon
            recorder.record(obs, cur_action, key_frame)

            # Omit the last item since the last item indicates episode end
            obs, reward, done, info = env.step(cur_action)
            
            
            env.render()
        # Save the episode data
        success = False
        if args.save:
            # The rollout runs to max_timesteps regardless of when the task is solved,
            # so trim the trailing frames where the action vector stops changing.
            episode_path = recorder.save(trim_still_tail=True)
            if args.critic:
                result = critic.critic_episode_from_frontview(episode_path, args.task_definition)
                success = result.success
                critic_record.record_episode(episode_path, result.success, result.reason)
        else:
            print("Episode not saved as per user request.")

        # Persistently show the running rollout progress bar after each rollout
        rollout_results.append(success)
        print_progress_bar(rollout_results)
    print()  # finish the in-place progress bar line
    if args.save:
            critic_record.save()
    print("End of rollout")
