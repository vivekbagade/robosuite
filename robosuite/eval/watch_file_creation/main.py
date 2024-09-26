# main.py

import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from google.cloud import storage


def load_hdf5(file_name: str, bucket_name: str) -> tuple:
    # Get the bucket name and file name from the event
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    # Download the file to a temporary location
    temp_file = f"/tmp/{os.path.basename(file_name)}"
    blob.download_to_filename(temp_file)

    with h5py.File(temp_file, "r") as root:
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        image_dict = dict()
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]
        action = root["/action"][()]

    return qpos, qvel, action, image_dict


def find_stopping_point(data: np.ndarray) -> int:
    steps, _ = data.shape
    for step in range(steps, -1, -1):
        if np.any(data[step:, :] != 0):
            return step

    return data.shape[0]


def sample_frames(qpos: np.ndarray, action: np.ndarray, num_samples: int) -> list:
    arr = qpos[:, 7]
    num_choices = num_samples // 3 - 1

    # When gripper is closed
    ones_indices = np.where(arr == 1)[0]

    if len(ones_indices) == 0:
        return random.choices([i for i in range(last_frame)], k=num_samples)

    # When gripper first closes
    last_zero_in_first_section = ones_indices[0] - 1

    # Before gripper first closes
    first_random_indices = random.choices([i for i in range(last_zero_in_first_section)], k=num_choices)

    # Sample when gripper remains closed
    random_one_index = random.choices(ones_indices, k=num_choices)

    last_one_index = ones_indices[-1]
    first_zero_in_last_section = np.argmax(arr[last_one_index + 1 :] == 0) + last_one_index + 1

    # Sample after gripper opens
    last_frame = find_stopping_point(action)
    last_random_indices = random.choices([i for i in range(first_zero_in_last_section, last_frame)], k=num_choices)

    steps = (
        [last_zero_in_first_section, first_zero_in_last_section, last_frame - 1]
        + random_one_index
        + first_random_indices
        + last_random_indices
    )
    steps = sorted(steps)
    return steps


def plot_steps(steps: list, camera: str, image_dict: dict, out_img_path: str) -> None:
    # Plot sample frames
    cols = 3
    rows = len(steps) // cols
    _, axes = plt.subplots(rows, cols, figsize=(8, 8))
    plt.figure()

    col = 0
    row = 0
    for cam_name, image_list in image_dict.items():
        if cam_name == camera:
            for step in steps:
                ax = axes[row, col]
                img = np.rot90(image_list[step], 2)
                ax.imshow(img)
                ax.axis("off")

                col += 1
                if col >= cols:
                    col = 0
                    row += 1

    plt.tight_layout()
    plt.savefig(out_img_path, format="jpg")


def gcs_trigger(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
        event (dict): The event payload.
        context (google.cloud.functions.Context): Metadata for the event.
    """
    out_img_path = "/tmp/out.jpg"
    file_name = event["name"]
    bucket_name = event["bucket"]
    print(f"Processing file {file_name}")
    qpos, qvel, action, image_dict = load_hdf5(file_name, bucket_name)
    print(f"Data loaded from {file_name}")
    steps = sample_frames(qpos=qpos, action=action, num_samples=9)
    plot_steps(steps=steps, camera="cam_1", image_dict=image_dict, out_img_path=out_img_path)
    print(f"Frames sampled and plotted")
    # file_name = event["name"]
    # bucket_name = event["bucket"]
    # print(f"File {file_name} created in bucket {bucket_name}.")
