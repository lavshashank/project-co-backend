from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import os
import time
import torch
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from basicsr.archs.rrdbnet_arch import RRDBNet


# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    model = None

# Load ESRGAN model
def load_esrgan_model(model_path):
    """Load the ESRGAN model."""
    try:
        esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        esrgan_model.load_state_dict(state_dict)
        esrgan_model.eval()
        return esrgan_model
    except Exception as e:
        print(f"Error loading ESRGAN model: {e}")
        return None

ESRGAN_MODEL_PATH = './ESRGAN/models/RRDB_ESRGAN_x4.pth'
upsampler = load_esrgan_model(ESRGAN_MODEL_PATH)

# Helper Functions
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_resolution(frame, model):
    """Enhance the resolution of a frame using ESRGAN."""
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            output_tensor = model(input_tensor)
        sr_frame = (output_tensor.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return sr_frame
    except Exception as e:
        print(f"Error during resolution enhancement: {e}")
        return frame

def generate_event_stream(file_path):
    """Send real-time updates via SSE."""
    while True:
        time.sleep(1)
        yield f"data:{realtime_data['collision_warning']}|{realtime_data['traffic_light']}\n\n"

realtime_data = {"collision_warning": "None", "traffic_light": "Unknown"}

def detect_lanes(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)

    # Draw lane lines on the frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def process_video_frame(frame, results):
    """Process video frame for collision and traffic light detection."""
    global realtime_data

    # Collision detection
    height, width = frame.shape[:2]
    roi_area = (width // 4, height // 2, 3 * width // 4, height)

    collision_boxes = []  # Store boxes of potential collision objects

    if results and results[0].boxes:
        try:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                if x1 < roi_area[2] and x2 > roi_area[0] and y2 > roi_area[1]:
                    collision_boxes.append((x1, y1, x2, y2))

            collision_percentage = (len(collision_boxes) / len(results[0].boxes.xyxy) * 100) if results[0].boxes else 0
            highest_risk_box = max(collision_boxes, key=lambda b: b[3] - b[1], default=None)  # Highest collision risk box
            realtime_data["collision_warning"] = (
                "HIGH COLLISION RISK" if highest_risk_box else "None"
            )
        except Exception as e:
            print(f"Error processing collision detection: {e}")

    # Draw bounding boxes and add collision warning
    try:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            color = (0, 255, 0)  # Default color for normal objects
            
            # Display collision warning only for the highest risk box
            if (x1, y1, x2, y2) == highest_risk_box:
                if highest_risk_box:
                    color = (0, 0, 255)  # Red for high collision risk
                    warning_text = "Collision Warning: High Risk"
                else:
                    warning_text = ""
            else:
                warning_text = ""

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Add collision warning text above the bounding box
            if warning_text:
                cv2.putText(frame, warning_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    except Exception as e:
        print(f"Error drawing bounding boxes or adding text: {e}")

    return frame

# Updated generate_frames function
def generate_frames():
    try:
        cap = cv2.VideoCapture()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance resolution
            if upsampler is not None:
                frame = enhance_resolution(frame, upsampler)

            # YOLO detection
            if model is not None:
                try:
                    results = model.predict(source=frame, save=False, conf=0.8)
                    frame = process_video_frame(frame, results)
                except Exception as e:
                    print(f"YOLO prediction error: {e}")

            # Encode and yield frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        cap.release()
    except Exception as e:
        print(f"Video feed error: {e}")

@app.route('/')
def homepage():
    return "Welcome to the homepage!"

# Routes
@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video uploads."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video_file = request.files['video']
    if video_file and allowed_file(video_file.filename):
        filename = secure_filename(video_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            video_file.save(file_path)
            return jsonify({'message': 'File uploaded successfully.', 'file_path': file_path}), 200
        except Exception as e:
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type.'}), 400

@app.route('/video_feed/<path:file_path>')
def video_feed(file_path):
    """Stream video with detection and resolution enhancement."""
    def generate_frames():
        try:
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Enhance resolution
                if upsampler is not None:
                    frame = enhance_resolution(frame, upsampler)

                # YOLO detection
                if model is not None:
                    try:
                        results = model.predict(source=frame, save=False, conf=0.8)
                        frame = results[0].plot()

                    except Exception as e:
                        print(f"YOLO prediction error: {e}")

                # Process frame
                process_video_frame(frame, results)

                # Encode and yield frame
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

            cap.release()
        except Exception as e:
            print(f"Video feed error: {e}")

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events/<path:file_path>')
def stream_events(file_path):
    """Provide real-time updates."""
    return Response(generate_event_stream(file_path), content_type='text/event-stream')

if __name__ == '__main__':
    app.run()
