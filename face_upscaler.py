import os
import cv2
import default_paths as dp
from upscaler.GPEN import GPEN
from upscaler.GFPGAN import GFPGAN
from upscaler.codeformer import CodeFormer
from upscaler.restoreformer import RestoreFormer

def gfpgan_runner(img, model):
    img = model.enhance(img)
    return img


def codeformer_runner(img, model):
    img = model.enhance(img, w=0.9)
    return img


def gpen_runner(img, model):
    img = model.enhance(img)
    return img


def restoreformer_runner(img, model):
    img = model.enhance(img)
    return img


supported_upscalers = {
    "CodeFormer": (dp.CODEFORMER_PATH, codeformer_runner),
    "GFPGANv1.4": (dp.GFPGAN_V14_PATH, gfpgan_runner),
    "GFPGANv1.3": (dp.GFPGAN_V13_PATH, gfpgan_runner),
    "GFPGANv1.2": (dp.GFPGAN_V12_PATH, gfpgan_runner),
    "GPEN-BFR-512": (dp.GPEN_BFR_512_PATH, gpen_runner),
    "GPEN-BFR-256": (dp.GPEN_BFR_256_PATH, gpen_runner),
    "RestoreFormer": (dp.RESTOREFORMER_PATH, gpen_runner),
}

cv2_upscalers = ["LANCZOS4", "CUBIC", "NEAREST"]

def get_available_upscalers_names():
    available = []
    for name, data in supported_upscalers.items():
        if os.path.exists(data[0]):
            available.append(name)
    return available


def load_face_upscaler(name='GFPGAN', provider=["CPUExecutionProvider"], session_options=None):
    assert name in get_available_upscalers_names() + cv2_upscalers, f"Face upscaler {name} unavailable."
    if name in supported_upscalers.keys():
        model_path, model_runner = supported_upscalers.get(name)
    if name == 'CodeFormer':
        model = CodeFormer(model_path=model_path, provider=provider, session_options=session_options)
    elif name.startswith('GFPGAN'):
        model = GFPGAN(model_path=model_path, provider=provider, session_options=session_options)
    elif name.startswith('GPEN'):
        model = GPEN(model_path=model_path, provider=provider, session_options=session_options)
    elif name == "RestoreFormer":
        model = RestoreFormer(model_path=model_path, provider=provider, session_options=session_options)
    elif name == 'LANCZOS4':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_LANCZOS4)
    elif name == 'CUBIC':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_CUBIC)
    elif name == 'NEAREST':
        model = None
        model_runner = lambda img, _: cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)
    else:
        model = None
    return (model, model_runner)