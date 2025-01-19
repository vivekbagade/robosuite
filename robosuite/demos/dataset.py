import tensorflow_datasets as tfds
from robosuite.recorder.dataset_builder import RobosuiteDatasetBuilder

dataset, info = tfds.load('RobosuiteDatasetBuilder', split='train', with_info=True)

