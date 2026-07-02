from absl import flags
import sys
import os
from critic import Critic
import time
from utils.utils import *

if __name__ == "__main__":
    flags.DEFINE_string("data_dir", "/act-data", "The dir containing training episodes, weights etc")
    flags.DEFINE_string("version", "1.0.0", "The version of the model to eval")
    flags.DEFINE_string("task", "PickPlaceCan", "The task to evaluate")
    flags.DEFINE_string("task_definition", "The robot should pick up the red can and place it in the right bin. The right bin has a silhouette of a can on it.", "The task definition to use for evaluation")
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)
    args = FLAGS

    episodes_dir = get_episodes_dir(args.data_dir, args.task, args.version)

    critic = Critic()
    episode_files = [f for f in os.listdir(episodes_dir) if f.endswith('.hdf5')]
    n = len(episode_files)
    n_success = 0
    if n == 0:
        print(f"No episodes found in {episodes_dir}. Please check the directory.")
        sys.exit(1)
    file = get_eval_result_file(args.data_dir, args.task, args.version)
    with open(file, "w") as f:
        f.write(f"Evaluating {n} episodes for task: {args.task}\n")
        f.write(f"Task Definition: {args.task_definition}\n\n")
        for episode_file in episode_files:
            try:
                episode_path = os.path.join(episodes_dir, episode_file)
                print(f"Evaluating episode: {episode_path}")
                result = critic.critic_episode_from_frontview_video(episode_path, args.task_definition)
                print(f"Evaluation Result: {result}")
                f.write(f"{episode_file}: {result.success} and {result.reason}\n")
                if result.success:
                    n_success += 1
            except Exception as e:
                print(f"Error evaluating episode {episode_file}: {e}")
                break
            time.sleep(100)
        f.write(f"\nTotal Successes: {n_success}/{n}\n")
    print(f"Evaluation complete. Success rate: {n_success}/{n}")
