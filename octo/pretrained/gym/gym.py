"""Use ACT policy to eval can pick and place.

"""
import pickle
import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper, GymWrapper
from octo.model.octo_model import OctoModel
from octo.utils import gym_wrappers
from functools import partial
import jax
from octo.utils.train_callbacks import supply_rng


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument(
        "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
    )
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch gripper control on gripper action")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch camera angle on gripper action")
    parser.add_argument("--controller", type=str, default="osc", help="Choice of controller. Can be 'ik' or 'osc'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()

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
    model = OctoModel.load_pretrained("/home/vivekbagade/octo-base")
    env = VisualizationWrapper(env, indicator_configs=None)
    env = GymWrapper(env)
    env = gym_wrappers.NormalizeProprio(env, model.dataset_statistics["bridge_dataset"])
    env = gym_wrappers.HistoryWrapper(env, horizon=2)
    env = gym_wrappers.RHCWrapper(env, exec_horizon=50)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    policy = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"]
        ),
    )

    # Reset the environment
    obs, _ = env.reset()
    task = model.create_tasks(texts="Pick up the red can")
    
    for t in range(400):
        norm_actions = model.sample_actions(jax.tree_map(lambda x: x[None], obs), task, rng=jax.random.PRNGKey(0))
        norm_actions = norm_actions[0]   # remove batch
        obs, _, _, truncated, _ = env.step(norm_actions)
        
        env.render()
    print("End of episode")
