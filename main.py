import os
import ultralytics
import supervision as sv
from ultralytics import YOLO
import numpy as np
import math

HOME = os.getcwd()

# from supervision.assets import download_assets, VideoAssets
SOURCE_VIDEO_PATH = f"{HOME}/videos/2025-01-03/left_angle_count_210.mp4" # left
# SOURCE_VIDEO_PATH = f"{HOME}/videos/2025-01-03/center_angle_count_213.MOV" # center
TARGET_VIDEO_PATH = f"{HOME}/out/videos/left_yolov8_sv_2025-01-03.mp4"

# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
print(video_info)

ultralytics.checks()
model = YOLO("yolov8x.pt")

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names

# the class names we have chosen
SELECTED_CLASS_NAMES = ['person', 'car', 'motorcycle', 'bus', 'truck']

# class ids matching the class names we have chosen
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name
    in SELECTED_CLASS_NAMES
]

# settings
LINE_START = sv.Point(0 + 50, 1500)
LINE_END = sv.Point(3840 - 50, 1500)

# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3)

byte_tracker.reset()

# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_color=sv.Color.BLACK)
# trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

tracks = {}

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    detections = byte_tracker.update_with_detections(detections)

    # compute movement
    for xyxy, t_id in zip(detections.xyxy, detections.tracker_id):
        center = ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)
        if t_id not in tracks:
            tracks[t_id] = {"center": center, "distance": 0.0}
        else:
            dist = math.dist(center, tracks[t_id]["center"])
            tracks[t_id]["distance"] += dist
            tracks[t_id]["center"] = center

    print(f"Frame {index}: {len(detections)} objects, total movement: {sum(t['distance'] for t in tracks.values()):.2f}")

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    line_zone.trigger(detections)
    return line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)