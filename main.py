import cv2
import torch
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracking algorithm

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLO11n model on the 'traffic.mp4' video
results = model("traffic.mp4", save=True, show=True, stream=True)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

# Video Input
video_path = 'traffic.mp4'
cap = cv2.VideoCapture(video_path)

# Output CSV storage
output_data = []
frame_count = 0

# Define Regions of Interest (ROI)
ROI_LEFT = 300
ROI_RIGHT = 1000
vehicle_count_left = 0
vehicle_count_right = 0
object_movements = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    results = model(frame)  # Run YOLO detection
    
    if results and isinstance(results, list):
        first_result = results[0]  # Extract first frame result
        detections = first_result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        classes = first_result.boxes.cls.cpu().numpy()  # Extract class labels
        confidences = first_result.boxes.conf.cpu().numpy()  # Extract confidence scores
    else:
        detections, classes, confidences = np.array([]), np.array([]), np.array([])
    
    # Convert to DeepSORT input format (x1, y1, x2, y2, confidence, class_id)
    dets = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections[i]
        conf = confidences[i]
        cls = classes[i]
        if int(cls) in [2, 3, 5, 7]:  # COCO classes for vehicle-related objects (car, bus, truck, train)
            dets.append(([x1, y1, x2, y2], conf, int(cls)))
    
    track_ids = tracker.update_tracks(dets, frame=frame)  # Track objects
    
    for track in track_ids:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = track.to_tlbr()
        track_id = track.track_id
        center_x = (x1 + x2) / 2

        # Determine movement trend
        if track_id in object_movements:
            prev_x = object_movements[track_id]
            movement_trend = "Right" if center_x > prev_x else "Left"
        else:
            movement_trend = "Middle"
        object_movements[track_id] = center_x

        # Check if object enters defined ROI regions
        zone = "None"
        if center_x < ROI_LEFT:
            vehicle_count_left += 1
            zone = "Left Lane"
        elif center_x > ROI_RIGHT:
            vehicle_count_right += 1
            zone = "Right Lane"
        
        output_data.append([frame_count, track_id, int(cls), movement_trend, zone])
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {int(track_id)}', (int(x1), int(y1)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display traffic statistics on screen
    cv2.putText(frame, f'Left Lane: {vehicle_count_left}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f'Right Lane: {vehicle_count_right}', (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ensure 'outputs/' directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save results to CSV only if data is available
cap.release()
cv2.destroyAllWindows()
if output_data:
    df = pd.DataFrame(output_data, columns=['Frame', 'Object_ID', 'Object_Class', 'Movement_Trend', 'Zone'])
    df.to_csv(os.path.join(output_dir, "tracking_data.csv"), index=False)
    print(f'Tracking data saved to {os.path.join(output_dir, "tracking_data.csv")}')
else:
    print("No tracking data recorded. CSV file was not created.")

# Traffic congestion classification
if vehicle_count_left + vehicle_count_right > 50:
    traffic_status = "Heavy Traffic"
elif vehicle_count_left + vehicle_count_right > 20:
    traffic_status = "Moderate Traffic"
else:
    traffic_status = "Light Traffic"

print(f'Tracking data saved. Traffic status: {traffic_status}')
