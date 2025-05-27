"""Use ACT policy to eval can pick and place.

"""
import pickle
import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import VisualizationWrapper
import jax
from octo.model.octo_model import OctoModel
from functools import partial
from octo.utils.train_callbacks import supply_rng
import cv2
import tkinter as tk
from PIL import Image, ImageTk

def get_image(obs, cam_name, resize_height, resize_width, old_image, show_image = False):
    img = cv2.resize(np.array(obs[cam_name + "_image"]), (resize_height, resize_width))
    img = cv2.rotate(img, cv2.ROTATE_180)
    if show_image:
        image = Image.fromarray(img)

        # Save the image as PNG
        image.save('output_image.png')
    if old_image.size == 0:
        return np.repeat(img[np.newaxis, :, :, :], 2, axis=0)
    else:
        return np.stack([old_image, img], axis=0)
# dataset = "RobosuiteDatasetBuilder"

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
    
    model = OctoModel.load_pretrained("/data/checkpoints/octo-finetune/1.1.0")

    stats = model.dataset_statistics['action']
    pre_process = lambda s_qpos: (s_qpos - stats['mean']) / stats['std']

    # Reset the environment
    obs = env.reset()
    # the supply_rng wrapper supplies a new random key to sample_actions every time it's called
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=model.dataset_statistics["action"],
        ),
    )
    task = model.create_tasks(texts=["Pick up the red can and place in the right place"])
    obs_step = dict()

    for t in range(600):
        obs_step['image_primary'] = np.array([np.array(obs["frontview_image"], dtype=np.uint8)])
        obs_step['image_wrist'] = np.array([np.array(obs["robot0_eye_in_hand_image"], dtype=np.uint8)])
        obs_step['proprio'] = np.array([np.array(obs['robot0_proprio-state'], dtype=np.float32)])
        obs_step['timestep_pad_mask'] = np.array([True])
        obs_step['task_completed'] = np.array([np.full(50, False)])
        obs_step['timestep'] = np.array([t])
        obs_step['pad_mask_dict'] = {
            'image_primary': np.array([True]),
            'image_wrist': np.array([True]),
            'timestep': np.array([True]),
            'proprio': np.array([True]),
        }

        model_observations = jax.tree_map(lambda x: x[None], obs_step)

        # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
        actions = model.sample_actions(
            model_observations, 
            task, 
            unnormalization_statistics=model.dataset_statistics["action"], 
            rng=jax.random.PRNGKey(0)
        )
        for action in actions[0]:
            obs, reward, done, info = env.step(action)
            env.render()
        
    print("End of episode")
