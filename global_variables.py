import os
from face_parsing import mask_regions
from utils.image import resolution_map
from face_upscaler import get_available_upscalers_names, cv2_upscalers
from face_analyser import single_face_detect_conditions, face_detect_conditions

DEFAULT_OUTPUT_PATH = os.getcwd()

MASK_BLUR_AMOUNT = 0.1
MASK_ERODE_AMOUNT = 0.15
MASK_REGIONS_DEFAULT = ["Skin", "R-Eyebrow", "L-Eyebrow", "L-Eye", "R-Eye", "Nose", "Mouth", "L-Lip", "U-Lip"]
MASK_REGIONS = list(mask_regions.keys())

NSFW_DETECTOR = None

FACE_ENHANCER_LIST = ["NONE"]
FACE_ENHANCER_LIST.extend(get_available_upscalers_names())
FACE_ENHANCER_LIST.extend(cv2_upscalers)

RESOLUTIONS = list(resolution_map.keys())

SINGLE_FACE_DETECT_CONDITIONS = single_face_detect_conditions
FACE_DETECT_CONDITIONS = face_detect_conditions
DETECT_CONDITION = "best detection"
DETECT_SIZE = 640
DETECT_THRESHOLD = 0.6

NUM_OF_SRC_SPECIFIC = 10

MAX_THREADS = 2

VIDEO_QUALITY_LIST = ["poor", "low", "medium", "high", "best"]
VIDEO_QUALITY = "best"

AVERAGING_METHODS = ["mean", "median"]
AVERAGING_METHOD = "mean"