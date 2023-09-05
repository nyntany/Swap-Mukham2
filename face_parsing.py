import cv2
import onnxruntime
import numpy as np

mask_regions = {
    "Background":0,
    "Skin":1,
    "L-Eyebrow":2,
    "R-Eyebrow":3,
    "L-Eye":4,
    "R-Eye":5,
    "Eye-G":6,
    "L-Ear":7,
    "R-Ear":8,
    "Ear-R":9,
    "Nose":10,
    "Mouth":11,
    "U-Lip":12,
    "L-Lip":13,
    "Neck":14,
    "Neck-L":15,
    "Cloth":16,
    "Hair":17,
    "Hat":18
}


class FaceParser:
    def __init__(self, model_path=None, provider=['CPUExecutionProvider'], session_options=None):
        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=self.session_options, providers=provider)
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def parse(self, img, regions=[1,2,3,4,5,10,11,12,13]):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = (img - self.mean) / self.std
        img = np.expand_dims(img.transpose((2, 0, 1)), axis=0).astype(np.float32)

        out = self.session.run(None, {'input':img})[0]
        out = out.squeeze(0).argmax(0)
        out = np.isin(out, regions).astype('float32')

        return out.clip(0, 1)


def mask_regions_to_list(values):
    out_ids = []
    for value in values:
        if value in mask_regions.keys():
            out_ids.append(mask_regions.get(value))
    return out_ids