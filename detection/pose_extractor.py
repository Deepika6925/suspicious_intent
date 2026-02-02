import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import os

MODEL_PATH = "models/pose_landmarker_full.task"
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)

def extract_pose(frame):
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = pose_landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return []
    return np.array([(lm.x, lm.y, lm.z) for lm in result.pose_landmarks[0]])
