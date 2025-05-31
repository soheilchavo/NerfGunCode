import os
import sys
import argparse
import glob
import time
import threading
from queue import Queue

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
parser.add_argument('--skip-frames', type=int, default=3, help='Process every Nth frame (default: 3)')
parser.add_argument('--buffer-size', type=int, default=1, help='Camera buffer size (default: 1)')
args = parser.parse_args()

# ---------------------- Parse Inputs ----------------------
model_path = args.model
img_source = args.source
conf_thresh = args.thresh
resW, resH = map(int, args.resolution.split('x'))
record = args.record
nogui = args.nogui
frame_skip = args.skip_frames
buffer_size = args.buffer_size

if not os.path.exists(model_path):
    print(f'ERROR: Model not found at {model_path}')
    sys.exit(0)

# Load model with optimizations
model = YOLO(model_path, task='detect')
labels = model.names  # Keep original class names

img_exts = ['.jpg','.jpeg','.png','.bmp']
vid_exts = ['.avi','.mov','.mp4','.mkv']

# Global variables for threading
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)
latest_detections = []
processing_active = True

def capture_frames(cap):
    """Separate thread for frame capture"""
    global processing_active
    while processing_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Keep queue small to avoid lag
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Skip frame if queue is full
            pass

def process_frames():
    """Separate thread for YOLO inference"""
    global processing_active, latest_detections
    
    while processing_active:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_resized = cv2.resize(frame, (resW, resH))
            
            # Run inference with optimizations
            results = model(frame_resized, 
                          verbose=False, 
                          imgsz=320,  # Keep small inference size
                          conf=conf_thresh,  # Filter low confidence early
                          classes=[0],  # Only detect person class
                          device='cpu')  # Explicit CPU usage
            
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = box.conf.item()
                    if conf >= conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy.flatten().tolist())
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'conf': conf
                        })
            
            latest_detections = detections
            
            if not result_queue.full():
                result_queue.put((frame_resized, detections))

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
    
    # Optimize camera settings for Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)  # Reduce buffer lag
    
    # Additional optimizations for USB cameras
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Reduce auto-exposure
    
else:
    print(f'Invalid source: {img_source}')
    sys.exit(0)

# Setup video writer
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only supports video/camera sources.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15, (resW, resH))

# Start processing threads for camera/video
if source_type in ['usb', 'video']:
    capture_thread = threading.Thread(target=capture_frames, args=(cap,))
    process_thread = threading.Thread(target=process_frames)
    capture_thread.daemon = True
    process_thread.daemon = True
    capture_thread.start()
    process_thread.start()

# ---------------------- Main Loop ----------------------
frame_count = 0
fps_buffer = []
fps_avg_len = 30  # Reduced buffer for more responsive FPS display
img_count = 0
last_process_time = time.time()

while True:
    t_start = time.perf_counter()

    # Load next frame
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('Done processing images.')
            break
        frame = cv2.imread(imgs_list[img_count])
        frame = cv2.resize(frame, (resW, resH))
        img_count += 1
        
        # Process images normally (no threading needed)
        if frame_count % frame_skip == 0:
            results = model(frame, verbose=False, imgsz=320, conf=conf_thresh, classes=[0])
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    conf = box.conf.item()
                    x1, y1, x2, y2 = map(int, box.xyxy.flatten().tolist())
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf
                    })
        else:
            detections = latest_detections  # Use last detections
            
    else:
        # For video/camera, get processed results from thread
        if not result_queue.empty():
            frame, detections = result_queue.get()
        else:
            # If no new results, continue with last frame and detections
            continue
    
    frame_count += 1

    # Draw detections
    object_count = len(detections)
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['conf']
        
        label = f"person: {int(conf * 100)}%"
        color = (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # FPS calculation
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    fps_buffer.append(fps)
    if len(fps_buffer) > fps_avg_len:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer)

    # Display info
    if source_type in ['usb', 'video']:
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.putText(frame, f'People: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if not nogui:
        cv2.imshow("YOLOv8 Person Detection", frame)

    if record:
        recorder.write(frame)

    # Reduced waitKey for better responsiveness
    key = cv2.waitKey(1 if not nogui else 1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

# Cleanup
processing_active = False
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()

print(f"Final Average FPS: {avg_fps:.2f}")
