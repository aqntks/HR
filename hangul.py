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
    weights, images, img_size, conf_thres, iou_thres = 'weights/best.pt', 'data/images', 320, 0.25, 0.45

    # 디바이스 세팅
    device = select_device("cpu")  # 첫번째 gpu 사용
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
                lines = []
                # 영역 검출 항목 저장
                for *xyxy, conf, cls in reversed(det):
                    obj.append((xyxy, conf, names[int(cls)]))
                    obj.sort(key=lambda x: x[0][0])
                for s, conf, cls in obj:
                    lines.append((s, cls, conf))

                    # result = result + cls + " " + str(conf) + " "
                # print('\n\n', result)
                lines = remove_intersect_box(lines)

                for l in lines:
                    print(l[1], end='')
                print('\n')

        print("\n검출 속도: " + str(time_synchronized() - startT) + '\n')


# 검출 박스 상자의 겹친 비율
def compute_intersect_ratio(rect1, rect2):
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    x3, y3, x4, y4 = rect2[0], rect2[1], rect2[2], rect2[3]

    if x2 < x3: return 0
    if x1 > x4: return 0
    if y2 < y3: return 0
    if y1 > y4: return 0

    left_up_x = max(x1, x3)
    left_up_y = max(y1, y3)
    right_down_x = min(x2, x4)
    right_down_y = min(y2, y4)

    width = right_down_x - left_up_x
    height = right_down_y - left_up_y

    original = (y2 - y1) * (x2 - x1)
    intersect = width * height

    ratio = int(intersect / original * 100)

    return ratio


# 겹친 상자 제거 (30% 이상)
def remove_intersect_box(lines):
    i, line = 0, lines.copy()
    while True:
        if i > len(line) - 2: break
        if compute_intersect_ratio(line[i][0], line[i+1][0]) > 30:
            lose = i if line[i][2] < line[i+1][2] else i+1
            del line[lose]
        else: i += 1

    return line

main()