from utils.utils import get_episodes_dir, get_eval_result_file
import os
import json

class EpisodeData:
    def __init__(self, episode_path: str, success: bool, reason: str):
        self.episode_path = episode_path
        self.success = success
        self.reason = reason

    def to_dict(self):
        return {
            "episode_path": self.episode_path,
            "success": self.success,
            "reason": self.reason
        }

class CriticRecord:
    def __init__(self, data_dir: str, task: str, version: str, task_definition: str):
        self.data_dir = data_dir
        self.task = task
        self.version = version
        self.task_definition = task_definition
        self.episodes_dir = get_episodes_dir(data_dir, task, version)
        self.eval_result_file = get_eval_result_file(data_dir, task, version)
        self.episode_data = []
        self.successful_episode_paths = []

    def save(self):
        success_percentage = len(self.successful_episode_paths) / len(self.episode_data) * 100 if self.episode_data else 0
        # Sort episode_data by episode_path
        self.episode_data.sort(key=lambda x: x.episode_path)
        # Write success_pecentage, episode_data, and successful_episode_paths to the eval_result_file as JSON
        with open(self.eval_result_file, 'w') as f:
            json.dump({
                "success_percentage": success_percentage,
                "episode_data": [data.to_dict() for data in self.episode_data],
                "successful_episode_paths": self.successful_episode_paths,
                "task_definition": self.task_definition
            }, f, indent=4)
        print(f"Evaluation results saved to {self.eval_result_file}")

    def record_episode(self, episode_path: str, success: bool, reason: str):
        episode_data = EpisodeData(episode_path, success, reason)
        self.episode_data.append(episode_data)
        if success:
            self.successful_episode_paths.append(episode_data.episode_path)
        return episode_data

    def read_successful_episodes(self):
        if not os.path.exists(self.eval_result_file):
            print(f"No evaluation results found at {self.eval_result_file}")
            return []
        with open(self.eval_result_file, 'r') as f:
            data = json.load(f)
        return data.get('successful_episode_paths', [])

        
        