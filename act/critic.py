
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
        except Exception as e:
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
        with h5py.File(episode_path, 'r') as root:
            key_frames = root['/observations/key_frame'][()]
            # Only keep the first 10 True key frames
            num_true_frames = 0
            limit = 10
            for i in range(len(key_frames)):
                if key_frames[i] == True:
                    num_true_frames += 1
                    if num_true_frames >= limit:
                        key_frames[i+1:] = False
                        break
            image_dict = dict()
            final_images = dict()
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
                final_images[cam_name] = [image_dict[cam_name][i] for i in range(len(key_frames)) if key_frames[i] == True]
        return final_images
    
    def critic_episode(self, episode_path, task_definition):
        # Extract key frames from the episode
        key_frames = self.extract_key_frames(episode_path)
        cameras = list(key_frames.keys())
        
        if len(key_frames) == 0 or len(key_frames[cameras[0]]) == 0:
            raise ValueError("No key frames found in the episode.")
        
        contents=[
            "You are a critic for robotic episodes. Your task is to evaluate the if the robot completed a task successfully or not based on the task definition and images provided.",
            "The most important part of your task is to make sure to respond with a json output with the following keys: 'success' (boolean), 'reason' (string).",
            f"Task Definition: {task_definition}\n",
            f"The images provided are from {len(cameras)} perpective(s) of the episode.",
            f"Each perspective contains {len(key_frames[cameras[0]])} images in a chronological order representing the robot trying to complete the task.",
        ]
        for camera, images in key_frames.items():
            contents.append(f"Here are the images from the camera {camera}:\n")
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
            model="gemini-2.5-flash",
            contents=contents,
        )
        
        return success(response.text)
    
# critic = Critic()
# print(critic.critic_episode("/act-data/PickPlaceCan/episodes/1.1.0-sim/episode_1.hdf5", "The robot should pick up the red can and place it in the right bin. The right bin has a silhouette of a can on it."))