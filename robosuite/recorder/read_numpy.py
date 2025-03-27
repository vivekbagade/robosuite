import numpy as np
import glob

def episode_len(episode_path):
    data = np.load(episode_path, allow_pickle=True)
    print(episode_path)
    print(f"Full length: {len(data)}")

    for i in range(len(data) - 1, -1, -1):
        if np.all(data[i]['proprio'] != 0.0):
            print(f"Non zero data len: {i}")
            return
    return len(data)

def trim_episode(target_len, episode_path):
    data = np.load(episode_path, allow_pickle=True)
    if len(data) <= target_len:
        return
    print(f"trimming {episode_path}")
    data = data[:target_len]
    new_path = episode_path.replace("train", "train_bak")
    print(f"saving {new_path}")
    np.save(new_path, data)

        


episode_paths = glob.glob("/data/episodes/train_bak/episode_*.npy")
for ep in episode_paths:
    #trim_episode(600, ep)
    episode_len(ep)

    
