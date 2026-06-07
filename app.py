import cv2
from ultralytics import YOLO
import time
import numpy as np
import threading
from flask import Flask, Response, render_template, jsonify, request
import queue
import base64

# --- Configuration ---
# IMPORTANT: Replace 'yolov8s.pt' with the actual path to your model file.
# Make sure the model file is accessible by the Flask application.
MODEL_PATH = 'yolov8s.pt' # Or 'runs/detect/train/weights/best.pt' if using your trained model
WEBCAM_SOURCE = 0
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.7

# --- Distance Estimation Parameters ---
KNOWN_OBJECT_DIMENSIONS = {
    'person': 1.50,
    'car': 1.80,
    'bicycle': 0.60,
    'truck': 2.50,
    'bus': 2.50,
    'table': 0.75,
    'chair': 0.90,
    'laptop': 0.35,
    'mouse': 0.07,
    'tv': 1.20,
    'bottle': 0.25,
    'backpack': 0.50,
    'bed': 1.50,
    'toilet': 0.70,
    'bowl': 0.15,
    'cup': 0.10,
    'refrigerator': 1.70,
    'motorcycle': 0.80,
    'airplane': 5.0, # Placeholder, depends on specific airplane type
    'train': 3.0, # Placeholder, depends on specific train car
    'boat': 2.0, # Placeholder, depends on boat type
    'traffic light': 0.7,
    'fire hydrant': 0.6,
    'stop sign': 0.75,
    'parking meter': 1.2,
    'bench': 0.45,
    'bird': 0.15,
    'cat': 0.3,
    'dog': 0.5,
    'horse': 1.6,
    'sheep': 0.8,
    'cow': 1.5,
    'elephant': 3.0,
    'bear': 1.5,
    'zebra': 1.5,
    'giraffe': 5.0,
    'umbrella': 0.9,
    'handbag': 0.3,
    'tie': 0.6,
    'suitcase': 0.6,
    'frisbee': 0.25,
    'skis': 1.7,
    'snowboard': 1.5,
    'sports ball': 0.22,
    'kite': 1.0,
    'baseball bat': 0.8,
    'baseball glove': 0.2,
    'skateboard': 0.8,
    'surfboard': 2.0,
    'tennis racket': 0.68,
    'wine glass': 0.2,
    'fork': 0.2,
    'knife': 0.25,
    'spoon': 0.2,
    'banana': 0.2,
    'apple': 0.08,
    'sandwich': 0.1,
    'orange': 0.07,
    'broccoli': 0.2,
    'carrot': 0.15,
    'hot dog': 0.15,
    'pizza': 0.3,
    'donut': 0.1,
    'cake': 0.2,
    'couch': 0.8,
    'potted plant': 0.5,
    'dining table': 0.75,
    'TV': 0.6, # Assuming TV height
    'remote': 0.15,
    'keyboard': 0.15,
    'cell phone': 0.15,
    'microwave': 0.3,
    'oven': 0.6,
    'toaster': 0.2,
    'sink': 0.4,
    'book': 0.2,
    'clock': 0.3,
    'vase': 0.25,
    'scissors': 0.15,
    'teddy bear': 0.4,
    'hair drier': 0.2,
    'toothbrush': 0.18
}

# Specifies whether to use width or height for distance calculation
KNOWN_DIMENSION_TYPE = {
    'person': 'height',
    'car': 'width',
    'bicycle': 'width',
    'truck': 'width',
    'bus': 'width',
    'table': 'height',
    'chair': 'height',
    'laptop': 'width',
    'mouse': 'width',
    'tv': 'height', # Changed to height for TV as it's often more consistent
    'bottle': 'height',
    'backpack': 'height',
    'bed': 'width',
    'toilet': 'height',
    'bowl': 'width',
    'cup': 'height',
    'refrigerator': 'height',
    'motorcycle': 'height',
    'airplane': 'width',
    'train': 'height',
    'boat': 'width',
    'traffic light': 'height',
    'fire hydrant': 'height',
    'stop sign': 'width',
    'parking meter': 'height',
    'bench': 'height',
    'bird': 'height',
    'cat': 'height',
    'dog': 'height',
    'horse': 'height',
    'sheep': 'height',
    'cow': 'height',
    'elephant': 'height',
    'bear': 'height',
    'zebra': 'height',
    'giraffe': 'height',
    'umbrella': 'height',
    'handbag': 'height',
    'tie': 'height',
    'suitcase': 'height',
    'frisbee': 'width',
    'skis': 'length', # Special case, might need custom logic or use width
    'snowboard': 'length', # Special case
    'sports ball': 'width',
    'kite': 'width',
    'baseball bat': 'length', # Special case
    'baseball glove': 'width',
    'skateboard': 'length', # Special case
    'surfboard': 'length', # Special case
    'tennis racket': 'length', # Special case
    'wine glass': 'height',
    'fork': 'height',
    'knife': 'height',
    'spoon': 'height',
    'banana': 'length',
    'apple': 'width',
    'sandwich': 'width',
    'orange': 'width',
    'broccoli': 'height',
    'carrot': 'length',
    'hot dog': 'length',
    'pizza': 'width',
    'donut': 'width',
    'cake': 'height',
    'couch': 'height',
    'potted plant': 'height',
    'dining table': 'height',
    'remote': 'length',
    'keyboard': 'width',
    'cell phone': 'height',
    'microwave': 'width',
    'oven': 'height',
    'toaster': 'height',
    'sink': 'width',
    'book': 'height',
    'clock': 'width',
    'vase': 'height',
    'scissors': 'length',
    'teddy bear': 'height',
    'hair drier': 'length',
    'toothbrush': 'length'
}

FOCAL_LENGTH_PIXELS = 300  # Calibrated focal length (in pixels) - Adjust as needed

# --- Audio Feedback Configuration ---
AUDIO_COOLDOWN_SECONDS = 5
STABILITY_DURATION_SECONDS = 2  # Object must be stable for 2 seconds
DISTANCE_CHANGE_THRESHOLD_METERS = 0.5
MIN_ANNOUNCE_DISTANCE = 0.1
MAX_ANNOUNCE_DISTANCE = 10.0

# --- Global State for Audio Logic (Thread-safe) ---
# Using a queue to pass audio announcements from the video processing thread to the Flask endpoint
audio_announcement_queue = queue.Queue(maxsize=1) # Only store the latest announcement

# Global state for object detection (shared between threads)
global_detection_state = {
    "last_announcement_time": 0,
    "last_announced_object": None,
    "last_announced_distance": 0.0,
    "last_announced_direction": None,
    "stable_object_class": None,
    "stable_object_distance": 0.0,
    "stable_object_direction": None,
    "stability_timer_start": 0,
    "current_fps": 0,
    "current_closest_object": "None",
    "current_closest_distance": "N/A",
    "current_closest_direction": "N/A"
}
state_lock = threading.Lock() # Lock for accessing global_detection_state

# --- Initialize Flask App ---
app = Flask(__name__)

# --- Load the YOLOv8 model ---
try:
    model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    # It's better to raise an exception or handle this gracefully in a real app
    # For this example, we'll let the app start but detection won't work.
    model = None

# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Processes a client-side camera frame and returns detection results."""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400

        current_time = time.time()
        
        # Perform object detection on the received frame
        if model is None:
            return jsonify({"error": "Model not initialized"}), 500

        results = model(frame, stream=True, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

        closest_object_data = None  # (class_name, distance_meters, direction_simple, direction_spoken)
        min_distance = float('inf')
        detected_objects = []

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = model.names

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = names[class_id]

                object_width_pixels = x2 - x1
                object_height_pixels = y2 - y1

                distance_meters = None
                
                # Calculate relative horizontal direction
                center_x = (x1 + x2) / 2
                if center_x < frame_width / 3:
                    direction_spoken = "on your left"
                    direction_simple = "Left"
                elif center_x > 2 * frame_width / 3:
                    direction_spoken = "on your right"
                    direction_simple = "Right"
                else:
                    direction_spoken = "in the center"
                    direction_simple = "Center"

                if class_name in KNOWN_OBJECT_DIMENSIONS and class_name in KNOWN_DIMENSION_TYPE:
                    known_dimension_real_world = KNOWN_OBJECT_DIMENSIONS[class_name]
                    dimension_type_for_calc = KNOWN_DIMENSION_TYPE[class_name]

                    if dimension_type_for_calc == 'height':
                        if object_height_pixels > 0:
                            distance_meters = (known_dimension_real_world * FOCAL_LENGTH_PIXELS) / object_height_pixels
                    elif dimension_type_for_calc == 'width':
                        if object_width_pixels > 0:
                            distance_meters = (known_dimension_real_world * FOCAL_LENGTH_PIXELS) / object_width_pixels

                    if distance_meters is not None:
                        if MIN_ANNOUNCE_DISTANCE <= distance_meters <= MAX_ANNOUNCE_DISTANCE:
                            if distance_meters < min_distance:
                                min_distance = distance_meters
                                closest_object_data = (class_name, distance_meters, direction_simple, direction_spoken)

                detected_objects.append({
                    "class": class_name,
                    "box": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "distance": f"{distance_meters:.2f}m" if distance_meters is not None else "N/A",
                    "direction": direction_simple
                })

        # --- Audio Announcement Logic (Stability and Cooldown) ---
        announcement_text = ""
        with state_lock:
            # Update global state for dashboard metrics
            global_detection_state["current_closest_object"] = closest_object_data[0] if closest_object_data else "None"
            global_detection_state["current_closest_distance"] = f"{closest_object_data[1]:.2f}m" if closest_object_data else "N/A"
            global_detection_state["current_closest_direction"] = closest_object_data[2] if closest_object_data else "N/A"

            if closest_object_data:
                current_closest_class, current_closest_distance, current_closest_dir_simple, current_closest_dir_spoken = closest_object_data

                object_changed = (current_closest_class != global_detection_state["stable_object_class"])
                distance_significantly_changed = (
                        abs(current_closest_distance - global_detection_state["stable_object_distance"]) > DISTANCE_CHANGE_THRESHOLD_METERS
                )
                direction_changed = (current_closest_dir_simple != global_detection_state.get("stable_object_direction"))

                if object_changed or distance_significantly_changed or direction_changed:
                    global_detection_state["stable_object_class"] = current_closest_class
                    global_detection_state["stable_object_distance"] = current_closest_distance
                    global_detection_state["stable_object_direction"] = current_closest_dir_simple
                    global_detection_state["stability_timer_start"] = current_time
                else:
                    if (current_time - global_detection_state["stability_timer_start"] >= STABILITY_DURATION_SECONDS and
                            current_time - global_detection_state["last_announcement_time"] >= AUDIO_COOLDOWN_SECONDS):

                        if MIN_ANNOUNCE_DISTANCE <= current_closest_distance <= MAX_ANNOUNCE_DISTANCE:
                            if (current_closest_class != global_detection_state["last_announced_object"] or
                                    abs(current_closest_distance - global_detection_state["last_announced_distance"]) > DISTANCE_CHANGE_THRESHOLD_METERS * 2 or
                                    current_closest_dir_simple != global_detection_state.get("last_announced_direction")):
                                announcement_text = (
                                    f"In front of you, there is a {current_closest_class} {current_closest_dir_spoken} "
                                    f"at {current_closest_distance:.2f} meters."
                                )

                                global_detection_state["last_announcement_time"] = current_time
                                global_detection_state["last_announced_object"] = current_closest_class
                                global_detection_state["last_announced_distance"] = current_closest_distance
                                global_detection_state["last_announced_direction"] = current_closest_dir_simple

            else:
                global_detection_state["stable_object_class"] = None
                global_detection_state["stable_object_distance"] = 0.0
                global_detection_state["stable_object_direction"] = None
                global_detection_state["last_announced_object"] = None
                global_detection_state["last_announced_distance"] = 0.0
                global_detection_state["last_announced_direction"] = None
                global_detection_state["last_announcement_time"] = 0

        # Calculate FPS
        fps = 0
        if "last_request_time" in global_detection_state:
            time_diff = current_time - global_detection_state["last_request_time"]
            if time_diff > 0:
                fps = int(1 / time_diff)
        global_detection_state["last_request_time"] = current_time
        global_detection_state["current_fps"] = fps

        return jsonify({
            "objects": detected_objects,
            "closest": {
                "class": closest_object_data[0] if closest_object_data else "None",
                "distance": f"{closest_object_data[1]:.2f}m" if closest_object_data else "N/A",
                "direction": closest_object_data[2] if closest_object_data else "N/A"
            },
            "announcement": announcement_text,
            "fps": fps
        })

    except Exception as e:
        print(f"Error in process_frame: {e}")
        return jsonify({"error": str(e)}), 500

# --- Main execution block ---
if __name__ == '__main__':
    # Port 7860 is required for Hugging Face Spaces deployment
    app.run(host='0.0.0.0', port=7860, debug=False, threaded=True)

