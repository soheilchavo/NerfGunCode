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

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default="320x240")  # Smaller default resolution
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--skip_frames', help='Skip every N frames for processing (process every Nth frame)',
                    default=2, type=int)
parser.add_argument('--imgsz', help='Inference image size (default: 320 for faster processing)',
                    default=320, type=int)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
skip_frames = args.skip_frames
inference_size = args.imgsz

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap with optimizations
model = YOLO(model_path, task='detect')
model.fuse()  # Fuse model layers for faster inference

# Enable optimizations
if hasattr(model.model, 'half'):
    model.model.half()  # Use FP16 for faster inference if supported

labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = 0
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = True  # Always resize for optimization
resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 15  # Reduced FPS for Pi
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Frame buffer for threading
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=2)

def capture_frames():
    """Thread function to capture frames"""
    global cap, source_type
    while True:
        if source_type == 'video' or source_type == 'usb':
            ret, frame = cap.read()
            if not ret:
                break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            ret = frame is not None
            if not ret:
                break
        
        if ret and not frame_queue.full():
            frame_queue.put(frame)

def process_frames():
    """Thread function to process frames with YOLO"""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            
            # Resize frame for inference (smaller = faster)
            inference_frame = cv2.resize(frame, (inference_size, inference_size))
            
            # Run inference
            results = model(inference_frame, verbose=False, conf=min_thresh)
            
            if not result_queue.full():
                result_queue.put((frame, results))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': 
        cap_arg = img_source
    elif source_type == 'usb': 
        cap_arg = usb_idx
    
    cap = cv2.VideoCapture(cap_arg)
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    config = cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)},
        controls={"FrameRate": 30}
    )
    cap.configure(config)
    cap.start()

# Set bounding box colors (reduced color list for efficiency)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 30  # Reduced buffer size
img_count = 0
frame_skip_counter = 0
last_results = None

# Start capture and processing threads for video sources
if source_type in ['video', 'usb', 'picamera']:
    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)
    capture_thread.start()
    process_thread.start()

# Begin inference loop
while True:
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count += 1
        
        # Resize and run inference
        frame = cv2.resize(frame, (resW, resH))
        inference_frame = cv2.resize(frame, (inference_size, inference_size))
        results = model(inference_frame, verbose=False, conf=min_thresh)
        
    else:  # Video sources with threading
        # Get processed results if available
        if not result_queue.empty():
            frame, results = result_queue.get()
            frame = cv2.resize(frame, (resW, resH))
            last_results = results
        elif last_results is not None:
            # Use last results if no new ones available
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame = cv2.resize(frame, (resW, resH))
            results = last_results
        else:
            continue

    # Extract results with optimized processing
    detections = results[0].boxes if results[0].boxes is not None else []
    object_count = 0

    # Optimized detection processing
    if len(detections) > 0:
        # Process all detections at once using vectorized operations
        xyxy_tensor = detections.xyxy.cpu().numpy()
        conf_tensor = detections.conf.cpu().numpy()
        cls_tensor = detections.cls.cpu().numpy().astype(int)
        
        # Scale coordinates back to display resolution
        scale_x = resW / inference_size
        scale_y = resH / inference_size
        
        for i in range(len(detections)):
            conf = conf_tensor[i]
            if conf > min_thresh:
                # Scale coordinates
                xmin = int(xyxy_tensor[i][0] * scale_x)
                ymin = int(xyxy_tensor[i][1] * scale_y)
                xmax = int(xyxy_tensor[i][2] * scale_x)
                ymax = int(xyxy_tensor[i][3] * scale_y)
                
                classidx = cls_tensor[i]
                classname = labels[classidx]
                
                # Simplified drawing (less text processing)
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                label = f'{classname}: {int(conf*100)}%'
                cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                object_count += 1

    # Draw simplified info
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.1f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    cv2.putText(frame, f'Objects: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    
    # Display results
    cv2.imshow('YOLO detection results', frame)
    if record: 
        recorder.write(frame)

    # Handle user input
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(1)  # Reduced wait time
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)
    
    # Calculate FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    
    # Simplified FPS calculation
    frame_rate_buffer.append(frame_rate_calc)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
cv2.destroyAllWindows()
