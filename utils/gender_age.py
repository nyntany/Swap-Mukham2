import cv2
import numpy as np
import onnxruntime
from .face_alignment import norm_crop2

class GenderAge:
    def __init__(self, model_file=None, provider=['CPUExecutionProvider'], session_options=None):
        self.model_file = model_file
        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.model_file, sess_options=self.session_options, providers=provider)

    def predict(self, img, kps):
        aimg, matrix = norm_crop2(img, kps, 128)

        blob = cv2.resize(aimg, (62,62), interpolation=cv2.INTER_AREA)
        blob = np.expand_dims(blob, axis=0).astype('float32')

        _prob, _age = self.session.run(None, {'data':blob})
        prob = _prob[0][0][0]
        age = round(_age[0][0][0][0] * 100)
        gender = np.argmax(prob)

        return gender, age
