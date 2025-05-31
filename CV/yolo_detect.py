import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------- Argument Setup ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (.pt)')
parser.add_argument('--source', required=True, help='"usb0", "video.mp4", folder, or image')
parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default='320x240', help='Resolution WxH (default: 320x240)')
parser.add_argument('--record', action='store_true', help='Save video to demo1.avi')
parser.add_argument('--nogui', action='store_true', help='Run without GUI preview')
args = parser.parse_args()

# ---------------------- Parse Inputs ----------------------
model_path = args.model
img_source = args.source
conf_thresh = args.thresh
resW, resH = map(int, args.resolution.split('x'))
record = args.record
nogui = args.nogui

if not os.path.exists(model_path):
    print(f'ERROR: Model not found at {model_path}')
    sys.exit(0)

model = YOLO(model_path, task='detect')
labels = model.names  # class names

img_exts = ['.jpg','.jpeg','.png','.bmp']
vid_exts = ['.avi','.mov','.mp4','.mkv','.wmv']

# Detect source type
if os.path.isdir(img_source):
    source_type = 'folder'
    imgs_list = sorted(glob.glob(os.path.join(img_source, '*')))
    imgs_list = [img for img in imgs_list if os.path.splitext(img)[-1].lower() in img_exts]
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[-1].lower()
    if ext in img_exts:
        source_type = 'image'
        imgs_list = [img_source]
    elif ext in vid_exts:
        source_type = 'video'
        cap = cv2.VideoCapture(img_source)
    else:
        print(f'Unsupported file type: {ext}')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    cam_index = int(img_source[3:])
    cap = cv2.VideoCapture(cam_index)
    cap.set(3, resW)
    cap.set(4, resH)
else:
    print(f'Invalid source: {img_source}')
    sys.exit(0)

# Setup video writer
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supports video/camera sources.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ---------------------- Main Loop ----------------------
frame_skip = 2
frame_count = 0
fps_buffer = []
fps_avg_len = 100
img_count = 0

while True:
    t_start = time.perf_counter()

    # Load next frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Done processing images.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Stream ended or camera disconnected.')
            break

    frame = cv2.resize(frame, (resW, resH))
    frame_count += 1

    # Skip frames to improve speed
    if frame_count % frame_skip != 0:
        continue

    results = model(frame, verbose=False, imgsz=320)
    detections = results[0].boxes

    object_count = 0
    for box in detections:
        classidx = int(box.cls.item())
        if classidx != 0:  # Only detect "person"
            continue

        conf = box.conf.item()
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy.flatten().tolist())
        label = f"person: {int(conf * 100)}%"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        object_count += 1

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    if source_type in ['usb', 'video']:
        cv2.putText(frame, f'FPS: {avg_fps:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f'People detected: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not nogui:
        cv2.imshow("YOLOv8 Person Detection", frame)

    if record:
        recorder.write(frame)

    key = cv2.waitKey(1 if not nogui else 0)
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

# Cleanup
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

print(f"Avg FPS: {avg_fps:.2f}")
