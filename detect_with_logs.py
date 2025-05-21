import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
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
    save_conf = opt.save_conf
    view_img = opt.view_img
    classes = opt.classes
    agnostic_nms = opt.agnostic_nms
    augment = opt.augment

    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img size

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'detections_log.txt'
    if log_file.exists():
        log_file.unlink()  # clear previous run

    # Run inference
    t0 = time.time()
    for frame_id, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            p = Path(path)
            save_path = str(save_dir / p.name)
            s = ''
            timestamp = 0

            # Compute video timestamp
            if isinstance(dataset, LoadStreams) or (hasattr(vid_cap, 'get') and vid_cap is not None):
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else dataset.fps
                timestamp = frame_id / fps if fps else 0

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=[255, 0, 0], line_thickness=2)

                    # Save detection to log file
                    with open(log_file, 'a') as f:
                        f.write(f"{timestamp:.2f}s: {names[int(cls)]} ({conf:.2f})\n")

            # Save results
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0s)
                else:  # video
                    if vid_path != save_path:
                        vid_path = save_path
                        if vid_writer:
                            vid_writer.release()
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                    vid_writer.write(im0s)

    print(f"Done. Log saved to {log_file}")
    print(f"Inference time: {time.time() - t0:.2f}s")

if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best_yolov5s_detect.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/video/block.mp4', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=384, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='inference/video_output', help='save results to project/name')
    parser.add_argument('--name', default='result_vid', help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    print(opt)

    detect()
