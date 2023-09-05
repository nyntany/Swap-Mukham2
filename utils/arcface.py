# -*- coding: utf-8 -*-
# @Organization  : insightface.ai
# @Author        : Jia Guo
# @Time          : 2021-09-18
# @Function      :


import os
import cv2
import onnx
import onnxruntime
import numpy as np
import default_paths as dp
from .face_alignment import norm_crop2


class ArcFace:
    def __init__(self, model_file=None, provider=['CUDAExecutionProvider'], session_options=None):
        assert model_file is not None
        self.model_file = model_file
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            #print(nid, node.name)
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        if find_sub and find_mul:
            #mxnet arcface model
            input_mean = 0.0
            input_std = 1.0
        else:
            input_mean = 127.5
            input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        #print('input mean and std:', self.input_mean, self.input_std)
        self.session_options = session_options
        if self.session_options is None:
            self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.model_file, providers=provider, sess_options=self.session_options)
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = []
        for out in outputs:
            output_names.append(out.name)
        self.input_name = input_name
        self.output_names = output_names
        assert len(self.output_names)==1
        self.output_shape = outputs[0].shape

    def prepare(self, ctx_id, **kwargs):
        if ctx_id<0:
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, kps):
        aimg, matrix = norm_crop2(img, landmark=kps, image_size=self.input_size[0])
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size

        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        blob = (batch_data - self.input_mean) / self.input_std
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out
