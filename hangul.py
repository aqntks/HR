import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import time
import re
import argparse


def main():
    weights, images, img_size, conf_thres, iou_thres = 'weights/hangul_0622.pt', 'data/images/hangul', 320, 0.01, 0.45

    # 디바이스 세팅
    device = select_device("0")  # 첫번째 gpu 사용
    half = device.type != 'cpu'  # gpu + cpu 섞어서 사용

    # 모델 로드
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if half:
        model.half()

    # 데이터 세팅
    dataset = LoadImages(images, img_size=img_size, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
    for path, img, im0s, vid_cap in dataset:
        startT = time_synchronized()
        # 이미지 정규화
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론 & NMS 적용    
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

        # 검출 값 처리
        for i, det in enumerate(pred):
            if len(det):
                result, obj, det[:, :4] = '', [], scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # 영역 검출 항목 저장
                for *xyxy, conf, cls in reversed(det):
                    obj.append((xyxy[0], conf, names[int(cls)]))
                    obj.sort(key=lambda x: x[0])
                for s, conf, cls in obj:
                    result = result + cls + " " + str(conf) + " "
                print('\n\n', result)

        print("\n검출 속도: " + str(time_synchronized() - startT) + '\n')