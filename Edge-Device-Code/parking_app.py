import streamlit as st
import cv2
import depthai as dai
import numpy as np
import json
import time
import threading
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from decimal import Decimal
from utils import decode_yolov8, get_aws_table, update_aws_table

# --- CONFIGURATION ---
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "parking_model.blob")
CONFIG_FILE = os.path.join(BASE_DIR, "parking_config.json")
DYNAMO_TABLE_NAME = "ParkingData"
AWS_REGION = "us-east-1"
CONFIDENCE_THRESHOLD = 0.3  # Lowered for screen demo (glare/reflection)
IOU_THRESHOLD = 0.5

# (decode_yolov8 helper function is imported from utils.py)

# --- SYSTEM CLASS ---
@st.cache_resource
class ParkingSystem:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.config = self.load_config()
        self.running = True
        self.detections = [] 
        self.last_aws_time = 0
        self.aws_status = "Waiting..."
        
        # Connect AWS
        self.table = get_aws_table(AWS_REGION, DYNAMO_TABLE_NAME)
        if self.table is not None:
            self.aws_status = "AWS Connected"
        else:
            self.aws_status = "AWS Connection Offline"

        # Start Camera Thread
        self.thread = threading.Thread(target=self.run_oak_d)
        self.thread.daemon = True
        self.thread.start()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"spots": [], "entrance": {"x": 320, "y": 480}, "rate": 0.05}

    def save_config(self, new_config):
        self.config = new_config
        with open(CONFIG_FILE, 'w') as f:
            json.dump(new_config, f)

    def run_oak_d(self):
        pipeline = dai.Pipeline()
        camRgb = pipeline.create(dai.node.ColorCamera)
        camRgb.setPreviewSize(640, 640)
        camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        camRgb.setInterleaved(False)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(MODEL_PATH)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        xoutNN = pipeline.create(dai.node.XLinkOut)
        xoutNN.setStreamName("nn")

        camRgb.preview.link(nn.input)
        camRgb.preview.link(xoutRgb.input)
        nn.out.link(xoutNN.input)

        with dai.Device(pipeline) as device:
            qRgb = device.getOutputQueue("rgb", 4, False)
            qDet = device.getOutputQueue("nn", 4, False)
            
            while self.running:
                inRgb = qRgb.get()
                inDet = qDet.get()
                
                # 1. Get Frame
                frame_cv = inRgb.getCvFrame()
                frame_resized = cv2.resize(frame_cv, (640, 480))
                
                # 2. Get Detections
                layer = inDet.getFirstLayerFp16()
                self.detections = decode_yolov8(layer, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
                
                # 3. Check Spots Intersection
                status_batch = []
                if self.config['spots']:
                    for spot in self.config['spots']:
                        is_occupied = False
                        
                        # Check if any car center is inside this spot
                        for det in self.detections:
                            cx = (det[0] * 640) + (det[2] * 640 / 2)
                            cy = (det[1] * 480) + (det[3] * 480 / 2)
                            
                            if (spot['x'] < cx < spot['x'] + spot['w'] and 
                                spot['y'] < cy < spot['y'] + spot['h']):
                                is_occupied = True
                                break
                        
                        # Calculate Distance
                        dist = 0
                        if self.config.get('entrance'):
                            dist = np.sqrt((spot['x'] - self.config['entrance']['x'])**2 + 
                                           (spot['y'] - self.config['entrance']['y'])**2)
                            dist = int(dist * 0.05) 

                        status_batch.append({
                            'SpotID': spot['id'],
                            'Status': 'Occupied' if is_occupied else 'Available',
                            'Distance': Decimal(str(dist)),
                            'Rate': Decimal(str(self.config.get('rate', 0.05)))
                        })

                # 4. AWS Update (Every 3s)
                if time.time() - self.last_aws_time > 3 and status_batch:
                    threading.Thread(target=self.update_aws, args=(status_batch,)).start()
                    self.last_aws_time = time.time()

                with self.lock:
                    # Convert to RGB for Streamlit
                    self.frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                time.sleep(0.01)

    def update_aws(self, data):
        success = update_aws_table(self.table, data)
        if success:
            self.aws_status = f"Last Update: {time.strftime('%H:%M:%S')}"
        else:
            self.aws_status = "AWS sync failed"

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# --- APP UI START ---
st.set_page_config(page_title="Parking AI", layout="wide")
system = ParkingSystem()

st.title("Smart Parking System")

# Status Bar
col1, col2 = st.columns(2)
col1.metric("System Status", "Running")
col2.metric("DynamoDB Sync", system.aws_status)

# --- TABS FOR MODE SELECTION ---
tab1, tab2 = st.tabs(["Live Monitor", "Setup & Calibration"])

with tab2:
    st.header("Calibration")
    st.info("1. Point camera at the screen. 2. Click 'Capture'. 3. Draw boxes. 4. Save.")
    
    # Capture Button
    if st.button("Capture Current View for Setup"):
        frame = system.get_frame()
        if frame is not None:
            # Save frame to session state to freeze it
            st.session_state['frozen_frame'] = frame
            
    # Display Canvas if frame is captured
    if 'frozen_frame' in st.session_state:
        bg_img = Image.fromarray(st.session_state['frozen_frame'])
        
        # Load existing spots
        init_draw = {"objects": []}
        if "spots" in system.config:
            for s in system.config['spots']:
                init_draw["objects"].append({
                    "type": "rect", "left": s['x'], "top": s['y'], 
                    "width": s['w'], "height": s['h'],
                    "stroke": "#00FF00", "strokeWidth": 2, "fill": "rgba(0,255,0,0.2)"
                })
        
        # Canvas
        canvas = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",
            stroke_width=2, stroke_color="#00FF00",
            background_image=bg_img,
            height=480, width=640,
            drawing_mode="rect",
            initial_drawing=init_draw,
            key="canvas_parking"
        )
        
        if st.button("Save Configuration"):
            if canvas.json_data:
                new_spots = []
                for i, obj in enumerate(canvas.json_data["objects"]):
                    new_spots.append({
                        "id": f"A{i+1}",
                        "x": int(obj["left"]), "y": int(obj["top"]),
                        "w": int(obj["width"]), "h": int(obj["height"])
                    })
                cfg = system.config
                cfg["spots"] = new_spots
                system.save_config(cfg)
                st.success(f"Saved {len(new_spots)} spots!")
    else:
        st.warning("Click 'Capture Current View' to start drawing.")

with tab1:
    st.header("Live Feed")
    show_debug = st.checkbox("Show AI Detections (Red Boxes)", value=True)
    
    placeholder = st.empty()
    while True:
        frame = system.get_frame()
        if frame is not None:
            # Draw Overlays
            for spot in system.config.get('spots', []):
                color = (0, 255, 0) # Green default
                
                # Check occupancy for color
                for det in system.detections:
                    cx = (det[0] * 640) + (det[2] * 640 / 2)
                    cy = (det[1] * 480) + (det[3] * 480 / 2)
                    if (spot['x'] < cx < spot['x'] + spot['w'] and 
                        spot['y'] < cy < spot['y'] + spot['h']):
                        color = (255, 0, 0) # Red if occupied
                        break
                
                cv2.rectangle(frame, (spot['x'], spot['y']), 
                             (spot['x']+spot['w'], spot['y']+spot['h']), color, 2)
                cv2.putText(frame, spot['id'], (spot['x'], spot['y']-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if show_debug:
                for det in system.detections:
                    x = int(det[0] * 640)
                    y = int(det[1] * 480)
                    w = int(det[2] * 640)
                    h = int(det[3] * 480)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)

            placeholder.image(frame)
        time.sleep(0.1)