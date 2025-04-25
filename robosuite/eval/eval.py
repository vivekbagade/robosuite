import os
import h5py
import mediapy as media
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


def load_hdf5(dataset_path: str) -> tuple:
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
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


# TODO: test this
def classify_episode(img_path: str):
    with open(img_path, "rb") as f:
        image_data = f.read()

    messages = [{"role": "user", "content": "Describe the content of the attached image."}]
    response = openai.ChatCompletion.create(
        model="gpt-4-vision",
        messages=messages,
        files=[("image", ("image.jpg", image_data, "image/jpeg"))],
    )
    return response.choices[0].message.content


out_img_path = "out.jpg"
data_file = "./episode_1.hdf5"
qpos, qvel, action, image_dict = load_hdf5(dataset_path=data_file)
steps = sample_frames(qpos=qpos, action=action, num_samples=9)
plot_steps(steps=steps, camera="frontview", image_dict=image_dict, out_img_path=out_img_path)
print(classify_episode(out_img_path))
