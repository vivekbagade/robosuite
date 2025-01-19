from typing import Any, Iterator, Tuple
import tensorflow_datasets.public_api as tfds
import numpy as np
from absl import flags
import glob
import tensorflow_hub as hub
from rlds import rlds_types

DATASET_PATH = flags.DEFINE_string(
    "dataset_path", "/data/episodes/train", "location to store episodes"
)

class RobosuiteDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(32,),
                            dtype=np.float32,
                            doc='[joint pos cos][joint pos sin][joint vel][EEF XYZ quat][gripper_qpos][gripper_qvel]',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, delta pos',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path=DATASET_PATH + '/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _generate_rlds_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            steps = []
            terminal_step = len(data) - 1
            for i in range(len(data) - 1, -1, -1):
                if np.all(data[i]['step'] != 0.0):
                    break
                terminal_step = i
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([step['language_instruction']])[0].numpy()
 
                steps.append(
                    rlds_types.build_step(
                        observation={
                            'image': step['frontview'],
                            'wrist_image': step['robot0_eye_in_hand'],
                            'state': step['proprio'],
                        },
                        action=step['action'],
                        is_first=i==0,
                        is_last=i==terminal_step,
                        metadata={
                            'language_instruction': step['language_instruction'],
                            'language_embedding': language_embedding,
                        }
                    )
                )

            rlds_episode = rlds_types.build_episode(
                steps=steps,
                metadata={
                    "episode_metadata": {
                        'episode_id': episode_path,
                    },
                },
            )

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, rlds_episode

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _generate_rlds_example(sample)

