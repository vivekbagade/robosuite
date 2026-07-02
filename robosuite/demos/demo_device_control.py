"""Teleoperate robot with keyboard or SpaceMouse.

***Choose user input option with the --device argument***

Keyboard:
    We use the keyboard to control the end-effector of the robot.
    The keyboard provides 6-DoF control commands through various keys.
    The commands are mapped to joint velocities through an inverse kinematics
    solver from Bullet physics.

    Note:
        To run this script with macOS, you must run it with root access.

SpaceMouse:

    We use the SpaceMouse 3D mouse to control the end-effector of the robot.
    The mouse provides 6-DoF control commands. The commands are mapped to joint
    velocities through an inverse kinematics solver from Bullet physics.

    The two side buttons of SpaceMouse are used for controlling the grippers.

    SpaceMouse Wireless from 3Dconnexion: https://www.3dconnexion.com/spacemouse_wireless/en/
    We used the SpaceMouse Wireless in our experiments. The paper below used the same device
    to collect human demonstrations for imitation learning.

    Reinforcement and Imitation Learning for Diverse Visuomotor Skills
    Yuke Zhu, Ziyu Wang, Josh Merel, Andrei Rusu, Tom Erez, Serkan Cabi, Saran Tunyasuvunakool,
    János Kramár, Raia Hadsell, Nando de Freitas, Nicolas Heess
    RSS 2018

    Note:
        This current implementation only supports macOS (Linux support can be added).
        Download and install the driver before running the script:
            https://www.3dconnexion.com/service/drivers.html

Additionally, --pos_sensitivity and --rot_sensitivity provide relative gains for increasing / decreasing the user input
device sensitivity


***Choose controller with the --controller argument***

Choice of using either inverse kinematics controller (ik) or operational space controller (osc):
Main difference is that user inputs with ik's rotations are always taken relative to eef coordinate frame, whereas
    user inputs with osc's rotations are taken relative to global frame (i.e.: static / camera frame of reference).

    Notes:
        OSC also tends to be more computationally efficient since IK relies on the backend pybullet IK solver.


***Choose environment specifics with the following arguments***

    --environment: Task to perform, e.g.: "Lift", "TwoArmPegInHole", "NutAssembly", etc.

    --robots: Robot(s) with which to perform the task. Can be any in
        {"Panda", "Sawyer", "IIWA", "Jaco", "Kinova3", "UR5e", "Baxter"}. Note that the environments include sanity
        checks, such that a "TwoArm..." environment will only accept either a 2-tuple of robot names or a single
        bimanual robot name, according to the specified configuration (see below), and all other environments will
        only accept a single single-armed robot name

    --config: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies the robot
        configuration desired for the task. Options are {"bimanual", "single-arm-parallel", and "single-arm-opposed"}

            -"bimanual": Sets up the environment for a single bimanual robot. Expects a single bimanual robot name to
                be specified in the --robots argument

            -"single-arm-parallel": Sets up the environment such that two single-armed robots are stationed next to
                each other facing the same direction. Expects a 2-tuple of single-armed robot names to be specified
                in the --robots argument.

            -"single-arm-opposed": Sets up the environment such that two single-armed robots are stationed opposed from
                each other, facing each other from opposite directions. Expects a 2-tuple of single-armed robot names
                to be specified in the --robots argument.

    --arm: Exclusively applicable and only should be specified for "TwoArm..." environments. Specifies which of the
        multiple arm eef's to control. The other (passive) arm will remain stationary. Options are {"right", "left"}
        (from the point of view of the robot(s) facing against the viewer direction)

    --switch-on-grasp: Exclusively applicable and only should be specified for "TwoArm..." environments. If enabled,
        will switch the current arm being controlled every time the gripper input is pressed

    --toggle-camera-on-grasp: If enabled, gripper input presses will cycle through the available camera angles

Examples:

    For normal single-arm environment:
        $ python demo_device_control.py --environment PickPlaceCan --robots Sawyer --controller osc

    For two-arm bimanual environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Baxter --config bimanual --arm left --controller osc

    For two-arm multi single-arm robot environment:
        $ python demo_device_control.py --environment TwoArmLift --robots Sawyer Sawyer --config single-arm-parallel --controller osc


"""

import argparse
import signal
import sys
import threading

import numpy as np
from pynput import keyboard

import robosuite as suite
from robosuite import load_controller_config
from robosuite.recorder import Recorder, grasp_state_from_command
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

# Cameras recorded for each episode (also used as the env's camera_names).
CAMERA_NAMES = ["robot0_eye_in_hand", "frontview", "birdview"]

# Ignore contact-change key frames during this many initial steps (avoids
# spurious key frames while the scene settles).
COLLISION_INIT_TIME = 100


class EpisodeKeyboardControl:
    """Thread-safe keyboard state shared between the pynput listener and the main loop.

    During an episode SPACE starts recording, F randomly repositions the object(s),
    and ESC ends the episode. After an episode ends, the loop waits on a Y / N / R
    keypress to save, discard, or retry.
    """

    def __init__(self):
        self.record = False
        self.episode_done = False
        self.episode_active = False
        self.reposition = False
        self._waiting_for_save = False
        self._save_decision = None
        self._save_event = threading.Event()
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._on_press)

    def start(self):
        self._listener.start()

    def _on_press(self, key):
        try:
            with self._lock:
                # While waiting on a save decision, only Y / N / R are meaningful.
                if self._waiting_for_save and self._save_decision is None:
                    char = getattr(key, "char", None)
                    if char in ("y", "n", "r"):
                        self._save_decision = {"y": True, "n": False, "r": "r"}[char]
                        self._save_event.set()
                    return
                if key == keyboard.Key.space and self.episode_active:
                    print("Recording started...")
                    self.record = True
                elif key == keyboard.Key.esc and self.episode_active:
                    self.episode_done = True
                elif getattr(key, "char", None) == "f" and self.episode_active:
                    self.reposition = True
        except Exception as e:
            print(f"Keyboard exc {e}")

    def reset_flags(self):
        """Clear per-episode flags before a new (re)run."""
        with self._lock:
            self.record = False
            self.episode_done = False
            self.reposition = False

    def set_active(self, active):
        with self._lock:
            self.episode_active = active

    def is_done(self):
        with self._lock:
            return self.episode_done

    def should_record(self):
        with self._lock:
            return self.record

    def consume_reposition(self):
        """Return True once if F was pressed since the last check, then clear it."""
        with self._lock:
            if self.reposition:
                self.reposition = False
                return True
            return False

    def wait_for_save_decision(self):
        """Block until the user presses Y / N / R; returns True / False / 'r'."""
        self._save_event.clear()
        with self._lock:
            self._save_decision = None
            self._waiting_for_save = True
        self._save_event.wait()
        with self._lock:
            self._waiting_for_save = False
            return self._save_decision


def parse_args():
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
    parser.add_argument("--data-dir", type=str, default="/act-data", help="The root ACT data directory")
    parser.add_argument("--version", type=str, default="1.0.0", help="The version of the model to record for")
    parser.add_argument("--num-of-episodes", type=int, default=10, help="Number of episodes to collect")
    return parser.parse_args()


def resolve_controller_name(controller):
    """Map the --controller choice to a robosuite controller config name."""
    if controller == "ik":
        return "IK_POSE"
    if controller == "osc":
        return "OSC_POSE"
    raise ValueError("Unsupported controller specified. Must be either 'ik' or 'osc'!")


def make_env(args, controller_config):
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    # Multi-armed environments need an explicit configuration; single-armed ones
    # must not carry a --config value (downstream logic keys off args.config).
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names=CAMERA_NAMES,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )
    return VisualizationWrapper(env, indicator_configs=None)


def make_device(args, env):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
        return device
    if args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        return SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    raise Exception("Invalid device choice: choose either 'keyboard' or 'spacemouse'.")


def pad_action(action, action_dim, arm):
    """Fit the device action to the environment's action space.

    Pads the unused arm of a multi-arm action with zeros, or trims trailing
    dimensions when the environment has no gripper action space.
    """
    rem_action_dim = action_dim - action.size
    if rem_action_dim > 0:
        rem_action = np.zeros(rem_action_dim)
        if arm == "right":
            return np.concatenate([action, rem_action])
        if arm == "left":
            return np.concatenate([rem_action, action])
        print("Error: Unsupported arm specified -- must be either 'right' or 'left'! Got: {}".format(arm))
        return action
    if rem_action_dim < 0:
        return action[:action_dim]
    return action


def reposition_objects(env):
    """Randomly re-place the environment's collision object(s), e.g. the can.

    Draws fresh poses from the environment's placement initializer and writes
    them to each object's free joint, then forwards physics so the new poses
    take effect. Returns a refreshed observation.
    """
    placements = env.placement_initializer.sample()
    for obj_pos, obj_quat, obj in placements.values():
        if "visual" in obj.name.lower():
            env.sim.model.body_pos[env.obj_body_id[obj.name]] = obj_pos
            env.sim.model.body_quat[env.obj_body_id[obj.name]] = obj_quat
        else:
            env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
    env.sim.forward()
    return env._get_observations()


def run_episode(env, device, recorder, controls, args, obs, cam_id, num_cam):
    """Run one teleop episode until the user ends it.

    Returns the latest observation and (possibly updated) camera id.
    """
    controls.reset_flags()
    recorder.reset()
    current_ncon = env.sim.data.ncon
    env.render()

    last_grasp = 0
    device.start_control()
    cur_episode_len = 0
    controls.set_active(True)

    while not controls.is_done():
        # F randomly repositions the object(s) mid-episode; refresh obs and the
        # contact baseline so the move isn't logged as a spurious key frame.
        if controls.consume_reposition():
            obs = reposition_objects(env)
            current_ncon = env.sim.data.ncon
            print("Repositioned object(s).")
            env.render()

        # Active robot may change mid-episode via --switch-on-grasp.
        active_robot = env.robots[0] if args.config == "bimanual" else env.robots[args.arm == "left"]

        action, grasp = input2action(
            device=device, robot=active_robot, active_arm=args.arm, env_configuration=args.config
        )
        cur_episode_len += 1

        # A None action signals a device reset, so end the episode.
        if action is None:
            break

        # On a fresh grasp press (last < 0 < current), optionally switch the
        # controlled arm and/or cycle the viewing camera.
        if last_grasp < 0 < grasp:
            if args.switch_on_grasp:
                args.arm = "left" if args.arm == "right" else "right"
            if args.toggle_camera_on_grasp:
                cam_id = (cam_id + 1) % num_cam
                env.viewer.set_camera(camera_id=cam_id)
        last_grasp = grasp

        action = pad_action(action, env.action_dim, args.arm)

        # Record the current obs paired with the chosen action, flagging a key
        # frame whenever the contact count changes after the scene has settled.
        obs["grasp"] = grasp_state_from_command(grasp)
        key_frame = abs(current_ncon - env.sim.data.ncon) > 0 and cur_episode_len >= COLLISION_INIT_TIME
        if key_frame:
            current_ncon = env.sim.data.ncon
        if controls.should_record():
            recorder.record(obs, action, key_frame)

        obs, reward, done, info = env.step(action)
        env.render()

    controls.set_active(False)
    return obs, cam_id


def announce_attempt(attempt, saved_count, total):
    print(f"\n=== Attempt {attempt} | Saved {saved_count} / {total} ===")
    print("Press SPACE to start recording, F to randomly reposition object(s), ESC to end episode.")


def collect_episodes(env, device, recorder, controls, args):
    saved_count = 0
    attempt = 0
    while saved_count < args.num_of_episodes:
        attempt += 1
        announce_attempt(attempt, saved_count, args.num_of_episodes)

        # Reset the environment and snapshot the initial state for retries.
        obs = env.reset()
        initial_sim_state = env.sim.get_state()
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)

        while True:  # retry loop — R restores initial_sim_state and reruns
            obs, cam_id = run_episode(env, device, recorder, controls, args, obs, cam_id, num_cam)

            print(f"Attempt {attempt} done. Press Y to save, N to discard, R to retry with same positions.")
            decision = controls.wait_for_save_decision()

            if decision == "r":
                print("Retrying with same object positions...")
                env.reset()  # resets robot controllers and internal state
                env.sim.set_state(initial_sim_state)  # restore initial physics (robot + objects)
                env.sim.forward()  # propagate restored state before reading obs
                obs = env._get_observations()
                attempt += 1
                announce_attempt(attempt, saved_count, args.num_of_episodes)
                continue
            if decision:
                path = recorder.save()
                if path:
                    saved_count += 1
                    print(f"Saved to {path} ({saved_count}/{args.num_of_episodes})")
                    break
                print("Press R to retry with same positions, N to discard.")
                continue
            print("Episode discarded.")
            break

    print(f"\nCollected {saved_count} episodes in {attempt} attempts. Exiting.")


def main():
    args = parse_args()

    controls = EpisodeKeyboardControl()
    controls.start()

    controller_config = load_controller_config(default_controller=resolve_controller_name(args.controller))
    env = make_env(args, controller_config)

    # Pretty-print numpy float arrays.
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    device = make_device(args, env)
    recorder = Recorder(CAMERA_NAMES, 256, 256, 800, args.environment, args.data_dir, args.version)

    def shutdown(exit_code=0):
        # Free the GL/EGL render contexts while EGL is still initialized.
        # Skipping this lets the contexts be destroyed during interpreter
        # teardown, when EGL is already gone, producing noisy EGL_NOT_INITIALIZED
        # errors from MjRenderContext.__del__.
        env.close()
        sys.exit(exit_code)

    def handler(arg1, arg2):
        recorder.save()
        print("Exiting..")
        shutdown(0)

    signal.signal(signal.SIGINT, handler)

    try:
        collect_episodes(env, device, recorder, controls, args)
    finally:
        shutdown(0)


if __name__ == "__main__":
    main()
