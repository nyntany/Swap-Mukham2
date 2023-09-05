import cv2
import onnxruntime
import numpy as np
import threading
import time

# gfpgan converted to onnx
# using https://github.com/xuanandsix/GFPGAN-onnxruntime-demo
# same inference code for GFPGANv1.2, GFPGANv1.3, GFPGANv1.4

lock = threading.Lock()

class GFPGAN:
    def __init__(self, model_path="GFPGANv1.4.onnx", provider=["CPUExecutionProvider"], session_options=None):
        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=self.session_options, providers=provider)
        self.resolution = self.session.get_inputs()[0].shape[-2:]

    def preprocess(self, img):
        img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0).astype(np.float32)
        return img

    def postprocess(self, img):
        img = (img.transpose(1,2,0).clip(-1,1) + 1) * 0.5
        img = (img * 255)[:,:,::-1]
        img = img.clip(0, 255).astype('uint8')
        return img

    def enhance(self, img):
        img = self.preprocess(img)
        with lock:
            output = self.session.run(None, {'input':img})[0][0]
        output = self.postprocess(output)
        return output
