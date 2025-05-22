import os
import argparse
import sys
import time
import csv
from pathlib import Path
import datetime

import cv2
import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

def save_detection_log(csv_path, data):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Class", "Confidence"])
        writer.writerows(data)

@torch.no_grad()
def run(weights='yolov5s.pt', source='data/images', img_size=640, conf_thres=0.25, iou_thres=0.45,
        device='', output='inference/output', target_class='anomaly', csv_output='detection_log.csv'):

    source = str(source)
    save_dir = Path(output)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(img_size, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    model.warmup(imgsz=(1 if pt else dataset.bs, 3, *imgsz))

    # support multiple classes: comma-separated
    target_classes = [cls.strip().lower() for cls in target_class.split(',')]

    log_data = []

    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        if frame_idx % 50 == 0:
            print(f"Processing frame {frame_idx} ...")

        img = torch.from_numpy(img).to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # Video timestamp formatting: HH:MM:SS.mmm
        if vid_cap:
            frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            video_time_sec = frame_number / fps if fps > 0 else 0
            td = datetime.timedelta(seconds=video_time_sec)
            timestamp = str(td)[:-3]  # drop microseconds to milliseconds
        else:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:
                    class_name = names[int(cls)]
                    if class_name.lower() in target_classes:
                        log_data.append([timestamp, class_name, f"{float(conf):.3f}"])

    if not log_data:
        print("No detections found for target class(es).")
    else:
        save_detection_log(csv_output, log_data)
        print(f"\n✅ Detection log saved to {csv_output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--source', type=str, default='data/images')
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--device', default='')
    parser.add_argument('--output', type=str, default='inference/output')
    parser.add_argument('--target_class', type=str, default='anomaly', help="Comma separated classes to log")
    parser.add_argument('--csv_output', type=str, default='detection_log.csv')
    opt = parser.parse_args()

    run(**vars(opt))
