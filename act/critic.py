
import json
import tempfile
import time
from PIL import Image
import imageio
import io
import h5py
import numpy as np
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
                model="gemini-3.6-flash",
                contents=contents,
                config=self.gen_config,
            )
        except errors.ServerError:
            # retry one more time in case of server error
            time.sleep(5)
            return self.gem_client.models.generate_content(
                model="gemini-3.6-flash",
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
        return final_images
    
    def critic_episode_from_frontview(self, episode_path, task_definition):
        # Extract key frames from the episode
        key_frames = self.extract_key_frames(episode_path)
        images = key_frames.get('frontview', [])

        if len(key_frames) == 0 or len(images) == 0:
            raise ValueError("No key frames found in the episode.")

        contents=[
            "You are a critic for robotic episodes. Your task is to evaluate if the robot completed a task successfully based on the task definition and images provided.",
            "CRITICAL SUCCESS CRITERIA:",
            "1. The robot must pick up the red can from its initial location",
            "2. The can must be placed in the RIGHT BIN (front-right from this frontview camera)",
            "3. The RIGHT BIN is identified by: it has a silhouette/shadow of a can printed on its front face",
            "4. In the FINAL IMAGE, the red can must be clearly visible INSIDE the right bin",
            "5. The can must not be teetering on the edge - it should be stably placed inside",
            "Respond with ONLY a json object with keys: 'success' (boolean), 'reason' (string explaining what you observed).",
            f"Task Definition: {task_definition}\n",
            f"Below are {len(images)} images in chronological order. Focus especially on the FINAL image to verify the can is in the correct right bin.",
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

    # Number of frames actually recorded, as opposed to the episode_len the file
    # is padded out to. The recorder pads with zeros and a real action is never
    # zero on every dimension, so the last non-zero action marks the end of the
    # footage. Files without an action dataset fall back to ``default``.
    @staticmethod
    def _recorded_len(root, default):
        if 'action' not in root:
            return default
        nonzero = np.flatnonzero(np.abs(root['action'][()]).sum(axis=1))
        return int(nonzero[-1]) + 1 if len(nonzero) else default

    # Encode every recorded frame from one camera in an episode file into a
    # compressed H.264 mp4 and return the raw video bytes.
    #
    # The cut is the recorded length, deliberately not the file's real_len:
    # real_len drops the trailing run of unchanging actions, which is exactly the
    # hold window the rollouts append so the scene can settle -- the gripper
    # opening and the object dropping both land inside it. The dataset wants that
    # tail treated as padding; the Critic has to see it, or it rules on a video
    # that cuts on the frame the release is commanded.
    def extract_video(self, episode_path, cam_name='frontview', fps=20):
        with h5py.File(episode_path, 'r') as root:
            frames = root[f'/observations/images/{cam_name}'][()]
            frames = frames[:self._recorded_len(root, len(frames))]
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

    def critic_episode_from_frontview_video(self, episode_path, task_definition, goal_image_path='/media/vivekbagade/Elements/act-data/PickPlaceCan/goal_image.png'):
        video_bytes = self.extract_video(episode_path, cam_name='frontview', fps=10)

        # Read goal image
        with open(goal_image_path, 'rb') as f:
            goal_bytes = f.read()

        contents = [
            "You are a critic for robotic episodes. Your task is to evaluate if the robot completed a task successfully based on the task definition, a goal image, and a video provided.",
            "CRITICAL SUCCESS CRITERIA:",
            "1. The robot must pick up the red can from its initial location",
            "2. The can must be placed in the RIGHT BIN (front-right bin from this frontview camera perspective)",
            "3. The RIGHT BIN is identified by: it has a silhouette/shadow of a can printed on its front face",
            "4. At the END of the video, the red can must be clearly visible INSIDE the right bin, matching the goal image",
            "5. The can must not be teetering on the edge - it should be stably and clearly placed inside the correct bin",
            "6. Watch the entire video carefully - pay special attention to the final frames showing the placement",
            "Respond with ONLY a json object with keys: 'success' (boolean), 'reason' (string explaining your observation of the final placement).",
            f"Task Definition: {task_definition}\n",
            "GOAL IMAGE (desired final state - red can placed in the right bin):",
            types.Part.from_bytes(
                data=goal_bytes,
                mime_type='image/png',
            ),
            "\nVIDEO of the robot attempting the task (focus on final frames to verify placement matches the goal):",
            types.Part.from_bytes(
                data=video_bytes,
                mime_type='video/mp4',
            ),
        ]
        response = self._generate(contents)
        return success.from_response(response.text)
    
# critic = Critic()
# print(critic.critic_episode_from_frontview_video("/media/vivekbagade/Elements/act-data/PickPlaceCan/episodes/5.1.0-sim/episode_1.hdf5", "The robot should pick up the red can from the bin where it is initially located to a smaller bin on the right. There are multiple bins on the right. The correct bin has a silhouette of a can on it. In the last image, you should check that the can is visible in the correct bin."))
