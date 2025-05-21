import os
import argparse
import sys
import time
import csv
from pathlib import Path

import cv2
import torch
import numpy as np
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    source = opt.source
    weights = opt.weights
    imgsz = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    device = opt.device
    save_txt = opt.save_txt
    view_img = opt.view_img
    classes = opt.classes
    agnostic_nms = opt.agnostic_nms
    augment = opt.augment

    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half()

    # Set Dataloader
    if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'detections_log.csv'

    # Create CSV log file
    with open(log_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'class', 'confidence'])

    # Inference
    t0 = time.time()
    vid_path, vid_writer = None, None
    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=augment)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        for i, det in enumerate(pred):
            p = Path(path)
            timestamp = 0
            if hasattr(vid_cap, 'get') and vid_cap is not None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                timestamp = frame_id / fps if fps else 0

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                with open(log_file, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for *xyxy, conf, cls in reversed(det):
                        writer.writerow([f'{timestamp:.2f}', names[int(cls)], f'{conf:.2f}'])

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir / p.name), im0s)
                else:
                    if vid_path != str(save_dir / p.name):
                        vid_path = str(save_dir / p.name)
                        if vid_writer:
                            vid_writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))
                    vid_writer.write(im0s)

            if view_img:
                cv2.imshow(str(p), im0s)
                if cv2.waitKey(1) == ord('q'):
                    raise StopIteration

    if vid_writer:
        vid_writer.release()

    print(f"✅ Detection completed.\n📄 CSV Log saved at: {log_file}\n⏱ Total inference time: {time.time() - t0:.2f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best_yolov5s_detect.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/video/block.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default='inference/video_output', help='save results to project/name')
    parser.add_argument('--name', default='result_vid', help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    detect()
