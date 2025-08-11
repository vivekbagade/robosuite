
import json
from PIL import Image
import io
import h5py
from google import genai
from google.genai import types
import os

class success:
    def __init__(self, success: bool, reason: str):
        self.success = success
        self.reason = reason

    def __init__(self, response: str):
        try:
            # Remove leading prefix of ```json
            if response.startswith("```json"):
                response = response[8:].strip()
            # Remove trailing suffix of ```
            if response.endswith("```"):
                response = response[:-3].strip()
            data = json.loads(response)
            self.success = data.get("success", False)
            self.reason = data.get("reason", "No reason provided")
        except json.JSONDecodeError as e:
            self.success = False
            self.reason = f"Error parsing response: {str(e)}"

    def to_dict(self):
        return {
            "success": self.success,
            "reason": self.reason
        }
    
    def __str__(self):
        return f"Success: {self.success}, Reason: {self.reason}"

class Critic:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.gem_client = genai.Client(api_key=api_key)


    # Extract key frames from an episode file. This function reads the HDF5 file and retrieves the key frames
    # based on the 'observations/key_frame'.
    def extract_key_frames(self, episode_path):
        min_diff = 5
        last_key_frame = -min_diff
        with h5py.File(episode_path, 'r') as root:
            key_frames = root['/observations/key_frame'][()]
            # Only keep the first 10 True key frames
            num_frames = 0
            limit = 10
            final_key_frames = [False] * len(key_frames)
            for i in range(len(key_frames)):
                if key_frames[i] == True and (i - last_key_frame) >= min_diff:
                    last_key_frame = i
                    num_frames += 1
                    if num_frames <= limit:
                        final_key_frames[i] = True
            key_frames = final_key_frames
            # Set the last key frame to True always
            key_frames[-1] = True
            image_dict = dict()
            final_images = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                final_images[cam_name] = [image_dict[cam_name][i] for i in range(len(key_frames)) if key_frames[i] == True]
                # if cam_name == 'frontview':
                #     # Rotate the image 180 degrees
                #     final_images[cam_name] = [img[::-1, ::-1, :] for img in final_images[cam_name]]
        return final_images
    
    def critic_episode_from_frontview(self, episode_path, task_definition):
        # Extract key frames from the episode
        key_frames = self.extract_key_frames(episode_path)
        images = key_frames.get('frontview', [])

        if len(key_frames) == 0 or len(images) == 0:
            raise ValueError("No key frames found in the episode.")

        contents=[
            "You are a critic for robotic episodes. Your task is to evaluate the if the robot completed a task successfully or not based on the task definition and images provided.",
            "The most important part of your task is to make sure to respond with a json output with the following keys: 'success' (boolean), 'reason' (string).",
            f"Task Definition: {task_definition}\n",
            f"Below are {len(images)} images in a chronological order representing the robot trying to complete the task.",
        ]
        for i in range(len(images)):
            img = Image.fromarray(images[i], 'RGB')
            image_bytes = io.BytesIO()
            img.save(image_bytes, 'PNG')
            contents.append(types.Part.from_bytes(
                data=image_bytes.getvalue(),
                    mime_type='image/png',
                ))

        
        # Call the Gemini API to get the critic's evaluation
        response = self.gem_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=contents,
        )
        
        return success(response.text)
    
# critic = Critic()
# print(critic.critic_episode_from_frontview("/act-data/PickPlaceCan/episodes/1.0.0-sim/episode_19.hdf5", "The robot should pick up the red can from the bin where it is initially located to a smaller bin on the right. There are multiple bins on the right. The correct bin has a silhouette of a can on it. In the last image, you should check that the can is visible in the correct bin."))
#critic.extract_key_frames("/act-data/PickPlaceCan/episodes/1.0.0-sim/episode_19.hdf5")