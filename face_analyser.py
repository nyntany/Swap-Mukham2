import os
import cv2
import threading
import numpy as np
from tqdm import tqdm
import concurrent.futures
import default_paths as dp
from dataclasses import dataclass
from utils.arcface import ArcFace
from utils.gender_age import GenderAge
from utils.retinaface import RetinaFace

cache = {}

@dataclass
class Face:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray
    gender: int
    age: int

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

single_face_detect_conditions = [
    "best detection",
    "left most",
    "right most",
    "top most",
    "bottom most",
    "middle",
    "biggest",
    "smallest",
]

multi_face_detect_conditions = [
    "all face",
    "specific face",
    "age less than",
    "age greater than",
    "all male",
    "all female"
]

face_detect_conditions =  multi_face_detect_conditions + single_face_detect_conditions


def get_single_face(faces, method="best detection"):
    total_faces = len(faces)

    if total_faces == 0:
        return None

    if total_faces == 1:
        return faces[0]

    if method == "best detection":
        return sorted(faces, key=lambda face: face["det_score"])[-1]
    elif method == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif method == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif method == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif method == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif method == "middle":
        return sorted(faces, key=lambda face: (
                (face["bbox"][0] + face["bbox"][2]) / 2 - 0.5) ** 2 +
                ((face["bbox"][1] + face["bbox"][3]) / 2 - 0.5) ** 2)[len(faces) // 2]
    elif method == "biggest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[-1]
    elif method == "smallest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[0]

def filter_face_by_age(faces, age, method="age less than"):
    if method == "age less than":
        return [face for face in faces if face["age"] < age]
    elif method == "age greater than":
        return [face for face in faces if face["age"] > age]
    elif method == "age equals to":
        return [face for face in faces if face["age"] == age]

def cosine_distance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 1 - np.dot(a, b)

def is_similar_face(face1, face2, threshold=0.6):
    distance = cosine_distance(face1["embedding"], face2["embedding"])
    return distance < threshold


class AnalyseFace:
    def __init__(self, provider=["CPUExecutionProvider"], session_options=None):
        self.detector = RetinaFace(model_file=dp.RETINAFACE_PATH, provider=provider, session_options=session_options)
        self.recognizer = ArcFace(model_file=dp.ARCFACE_PATH, provider=provider, session_options=session_options)
        self.gender_age = GenderAge(model_file=dp.GENDERAGE_PATH, provider=provider, session_options=session_options)
        self.detect_condition = "best detection"
        self.detection_size = (640, 640)
        self.detection_threshold = 0.5

    def analyser(self, img, skip_task=[]):
        bboxes, kpss = self.detector.detect(img, input_size=self.detection_size, det_thresh=self.detection_threshold)
        faces = []
        for i in range(bboxes.shape[0]):
            feat, gender, age = None, None, None
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            if 'embedding' not in skip_task:
                feat = self.recognizer.get(img, kpss[i])
            if 'gender_age' not in skip_task:
                gender, age = self.gender_age.predict(img, kpss[i])
            face = Face(bbox=bbox, kps=kps, det_score=det_score, embedding=feat, gender=gender, age=age)
            faces.append(face)
        return faces

    def get_faces(self, image, scale=1., skip_task=[]):
        if isinstance(image, str):
            image = cv2.imread(image)

        faces = self.analyser(image, skip_task=skip_task)

        if scale != 1: # landmark-scale
            for i, face in enumerate(faces):
                landmark = face['kps']
                center = np.mean(landmark, axis=0)
                landmark = center + (landmark - center) * scale
                faces[i]['kps'] = landmark

        return faces

    def get_face(self, image, scale=1., skip_task=[]):
        faces = self.get_faces(image, scale=scale, skip_task=skip_task)
        return get_single_face(faces, method=self.detect_condition)

    def get_averaged_face(self, images, method="mean"):
        if not isinstance(images, list):
            images = [images]

        face = self.get_face(images[0], scale=1., skip_task=[])

        if len(images) > 1:
            embeddings = [face['embedding']]

            for image in images[1:]:
                face = self.get_face(image, scale=1., skip_task=[])
                embeddings.append(face['embedding'])

            if method == "mean":
                avg_embedding = np.mean(embeddings, axis=0)
            elif method == "median":
                avg_embedding = np.median(embeddings, axis=0)

            face['embedding'] = avg_embedding

        return face