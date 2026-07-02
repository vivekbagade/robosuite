
import json
import tempfile
import time
from PIL import Image
import imageio
import io
import h5py
from google import genai
from google.genai import types
from google.genai import errors
import os

class success:
    def __init__(self, success: bool, reason: str):
        self.success = success
        self.reason = reason

    @classmethod
    def from_response(cls, response: str):
        try:
            if response.startswith("```json"):
                response = response[8:].strip()
            if response.endswith("```"):
                response = response[:-3].strip()
            data = json.loads(response)
            return cls(data.get("success", False), data.get("reason", "No reason provided"))
        except json.JSONDecodeError as e:
            return cls(False, f"Error parsing response: {str(e)}")

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
        self.gen_config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type='application/json',
        )

    def _generate(self, contents):
        try:
            return self.gem_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=self.gen_config,
            )
        except errors.ServerError:
            # retry one more time in case of server error
            time.sleep(5)
            return self.gem_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=contents,
                config=self.gen_config,
            )


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

        response = self._generate(contents)
        return success.from_response(response.text)

    # Encode all frames from one camera in an episode file into a compressed
    # H.264 mp4 and return the raw video bytes.
    def extract_video(self, episode_path, cam_name='frontview', fps=20):
        with h5py.File(episode_path, 'r') as root:
            frames = root[f'/observations/images/{cam_name}'][()]
        if cam_name == 'frontview':
            # frontview frames are recorded upside down, flip vertically
            frames = frames[:, ::-1, :, :]
        fd, tmp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        try:
            with imageio.get_writer(tmp_path, format='ffmpeg', mode='I',
                                    fps=fps, codec='libx264') as writer:
                for frame in frames:
                    writer.append_data(frame)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            os.remove(tmp_path)

    def critic_episode_from_frontview_video(self, episode_path, task_definition):
        video_bytes = self.extract_video(episode_path, cam_name='frontview', fps=10)
        contents = [
            "You are a critic for robotic episodes. Your task is to evaluate the if the robot completed a task successfully or not based on the task definition and video provided.",
            "Make sure to respond with a json output with the following keys: 'success' (boolean), 'reason' (string).",
            f"Task Definition: {task_definition}\n",
            "Below is a video of the robot trying to complete the task.",
            types.Part.from_bytes(
                data=video_bytes,
                mime_type='video/mp4',
            ),
        ]
        response = self._generate(contents)
        return success.from_response(response.text)
    
# critic = Critic()
# print(critic.critic_episode_from_frontview_video("/media/vivekbagade/Elements/act-data/PickPlaceCan/episodes/5.1.0-sim/episode_1.hdf5", "The robot should pick up the red can from the bin where it is initially located to a smaller bin on the right. There are multiple bins on the right. The correct bin has a silhouette of a can on it. In the last image, you should check that the can is visible in the correct bin."))
