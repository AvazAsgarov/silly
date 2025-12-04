"""
Romantic Webcam Filter App 💕
A fun and playful app with silly filters for you and your girlfriend!
"""
import os
import sys

# Suppress TensorFlow / MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Prevent sys.excepthook errors from printing
sys.excepthook = lambda *args, **kwargs: None



import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="💕 Love Filter Cam 💕",
    page_icon="💕",
    layout="wide"
)

# Custom CSS for romantic theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffeef8 0%, #ffe0f0 100%);
    }
    h1 {
        color: #ff1493;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-shadow: 2px 2px 4px rgba(255,20,147,0.3);
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff69b4, #ff1493);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(255,20,147,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Title with hearts
st.markdown("# 💕 Love Filter Cam 💕")
st.markdown("### *Made with love for my amazing girlfriend Zhala* 💖")
st.markdown("---")

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Sidebar for filter controls
st.sidebar.markdown("## 🎨 Filter Controls")
st.sidebar.markdown("Toggle your favorite filters!")

filters = {
    "big_eyes": st.sidebar.checkbox("👀 Big Eyes", value=False),
    "funny_mouth": st.sidebar.checkbox("👄 Funny Mouth Stretch", value=False),
    "rainbow_hair": st.sidebar.checkbox("🌈 Rainbow Hair", value=False),
    "cartoon": st.sidebar.checkbox("🎨 Cartoon Effect", value=False),
    "pixelated": st.sidebar.checkbox("🟦 Pixelated Face", value=False),
    "silly_colors": st.sidebar.checkbox("🎭 Silly Colors", value=False),
    "hearts": st.sidebar.checkbox("💕 Floating Hearts", value=True),
}

# Intensity controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ Effect Intensity")
eye_scale = st.sidebar.slider("Eye Size", 1.0, 3.0, 1.5)
mouth_stretch = st.sidebar.slider("Mouth Stretch", 1.0, 2.0, 1.3)
pixel_size = st.sidebar.slider("Pixel Size", 5, 30, 15)


class VideoProcessor(VideoProcessorBase):
    """
    Main video processor that applies filters to each frame
    """
    
    def __init__(self):
        self.filters = filters
        self.eye_scale = eye_scale
        self.mouth_stretch = mouth_stretch
        self.pixel_size = pixel_size
        self.heart_positions = [(np.random.randint(50, 590), np.random.randint(50, 430)) for _ in range(10)]
        self.heart_speeds = [np.random.randint(1, 3) for _ in range(10)]
    
    def apply_big_eyes(self, frame, landmarks, h, w):
        """
        FILTER: Big Eyes Effect
        Enlarges the eyes using facial landmarks
        """
        try:
            # Left eye indices (approximate)
            left_eye = [33, 133, 160, 159, 158, 157, 173]
            right_eye = [362, 263, 387, 386, 385, 384, 398]
            
            for eye_indices in [left_eye, right_eye]:
                # Get eye center and points
                eye_points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                                      for i in eye_indices])
                
                if len(eye_points) > 0:
                    eye_center = eye_points.mean(axis=0).astype(int)
                    
                    # Create circular mask for eye region
                    radius = int(np.linalg.norm(eye_points[0] - eye_center) * self.eye_scale)
                    
                    # Extract and enlarge eye region
                    y1, y2 = max(0, eye_center[1] - radius), min(h, eye_center[1] + radius)
                    x1, x2 = max(0, eye_center[0] - radius), min(w, eye_center[0] + radius)
                    
                    if y2 > y1 and x2 > x1:
                        eye_region = frame[y1:y2, x1:x2]
                        enlarged = cv2.resize(eye_region, None, fx=1.3, fy=1.3, 
                                            interpolation=cv2.INTER_LINEAR)
                        
                        # Center the enlarged eye
                        eh, ew = enlarged.shape[:2]
                        start_y = max(0, (eh - (y2-y1)) // 2)
                        start_x = max(0, (ew - (x2-x1)) // 2)
                        
                        crop_h = min(eh - start_y, y2 - y1)
                        crop_w = min(ew - start_x, x2 - x1)
                        
                        frame[y1:y1+crop_h, x1:x1+crop_w] = enlarged[start_y:start_y+crop_h, 
                                                                     start_x:start_x+crop_w]
        except Exception as e:
            pass
        
        return frame
    
    def apply_funny_mouth(self, frame, landmarks, h, w):
        """
        FILTER: Funny Mouth Stretch
        Stretches the mouth horizontally for a funny effect
        """
        try:
            # Mouth indices
            mouth_indices = [61, 291, 0, 17, 39, 269]
            mouth_points = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                                    for i in mouth_indices])
            
            if len(mouth_points) > 0:
                mouth_center = mouth_points.mean(axis=0).astype(int)
                
                # Define mouth region
                radius = 40
                y1, y2 = max(0, mouth_center[1] - radius), min(h, mouth_center[1] + radius)
                x1, x2 = max(0, mouth_center[0] - radius), min(w, mouth_center[0] + radius)
                
                if y2 > y1 and x2 > x1:
                    mouth_region = frame[y1:y2, x1:x2]
                    stretched = cv2.resize(mouth_region, None, 
                                         fx=self.mouth_stretch, fy=1.0,
                                         interpolation=cv2.INTER_LINEAR)
                    
                    sh, sw = stretched.shape[:2]
                    start_x = max(0, (sw - (x2-x1)) // 2)
                    crop_w = min(sw - start_x, x2 - x1)
                    
                    frame[y1:y2, x1:x1+crop_w] = stretched[:y2-y1, start_x:start_x+crop_w]
        except Exception as e:
            pass
        
        return frame
    
    def apply_rainbow_hair(self, frame, landmarks, h, w):
        """
        FILTER: Rainbow Hair
        Adds colorful rainbow overlay to hair region
        """
        try:
            # Get forehead point
            forehead = landmarks[10]
            forehead_y = int(forehead.y * h)
            
            # Create rainbow gradient for hair region
            hair_region = frame[0:forehead_y, :]
            
            # Create rainbow gradient
            rainbow = np.zeros_like(hair_region)
            for i in range(forehead_y):
                hue = int((i / forehead_y) * 180)
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                rainbow[i, :] = color
            
            # Blend with original
            frame[0:forehead_y, :] = cv2.addWeighted(hair_region, 0.6, rainbow, 0.4, 0)
        except Exception as e:
            pass
        
        return frame
    
    def apply_cartoon_effect(self, frame):
        """
        FILTER: Cartoon/Toon Effect
        Makes the image look like a cartoon
        """
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        
        # Color quantization
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        
        # Combine
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges)
        
        return cartoon
    
    def apply_pixelated_face(self, frame, landmarks, h, w):
        """
        FILTER: Pixelated Face
        Creates a pixelated/mosaic effect on the face
        """
        try:
            # Get face bounding box
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            
            x1, x2 = max(0, min(x_coords) - 30), min(w, max(x_coords) + 30)
            y1, y2 = max(0, min(y_coords) - 50), min(h, max(y_coords) + 30)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size > 0:
                # Pixelate
                small = cv2.resize(face_region, 
                                  (face_region.shape[1] // self.pixel_size, 
                                   face_region.shape[0] // self.pixel_size),
                                  interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (face_region.shape[1], face_region.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                
                frame[y1:y2, x1:x2] = pixelated
        except Exception as e:
            pass
        
        return frame
    
    def apply_silly_colors(self, frame):
        """
        FILTER: Silly Colors (Inverted/Weird Colors)
        Inverts or shifts colors for a silly effect
        """
        # Convert to HSV and shift hue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + 90) % 180  # Shift hue
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def draw_floating_hearts(self, frame):
        """
        DECORATION: Floating Hearts
        Draws animated hearts floating on the screen
        """
        h, w = frame.shape[:2]
        
        # Update heart positions
        for i in range(len(self.heart_positions)):
            x, y = self.heart_positions[i]
            y -= self.heart_speeds[i]
            
            # Reset if heart goes off screen
            if y < 0:
                y = h
                x = np.random.randint(50, w - 50)
            
            self.heart_positions[i] = (x, y)
            
            # Draw heart
            heart_size = 20
            cv2.circle(frame, (x - heart_size//4, y), heart_size//2, (147, 20, 255), -1)
            cv2.circle(frame, (x + heart_size//4, y), heart_size//2, (147, 20, 255), -1)
            pts = np.array([[x, y + heart_size//2], 
                          [x - heart_size//2, y], 
                          [x + heart_size//2, y]], np.int32)
            cv2.fillPoly(frame, [pts], (147, 20, 255))
        
        return frame
    
    def recv(self, frame):
        """
        Main processing function called for each frame
        """
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        
        # Update filter states
        self.filters = filters
        self.eye_scale = eye_scale
        self.mouth_stretch = mouth_stretch
        self.pixel_size = pixel_size
        
        # Process face landmarks
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Apply filters based on selection
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            if self.filters["big_eyes"]:
                img = self.apply_big_eyes(img, landmarks, h, w)
            
            if self.filters["funny_mouth"]:
                img = self.apply_funny_mouth(img, landmarks, h, w)
            
            if self.filters["rainbow_hair"]:
                img = self.apply_rainbow_hair(img, landmarks, h, w)
            
            if self.filters["pixelated"]:
                img = self.apply_pixelated_face(img, landmarks, h, w)
        
        # Apply filters that don't need face detection
        if self.filters["cartoon"]:
            img = self.apply_cartoon_effect(img)
        
        if self.filters["silly_colors"]:
            img = self.apply_silly_colors(img)
        
        # Add floating hearts decoration
        if self.filters["hearts"]:
            img = self.draw_floating_hearts(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Main app layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📸 Live Camera Feed")
    
    # WebRTC streamer
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    webrtc_ctx = webrtc_streamer(
        key="love-filter-cam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("### 💝 Love Notes")
    st.info("👉 Toggle filters in the sidebar!")
    st.success("📸 Adjust intensity sliders for more fun!")
    st.warning("💕 Hearts are enabled by default to make everything more romantic!")
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Allow camera access when prompted
    2. Select filters from the sidebar
    3. Adjust intensity sliders
    4. Have fun and smile! 😊
    """)
    
    st.markdown("---")
    st.markdown("### 💌 Special Message")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #ff69b4, #ff1493); 
                padding: 20px; border-radius: 15px; color: white; text-align: center;'>
        <h3>Made with 💕 just for you!</h3>
        <p>Every silly filter is a reminder of how much fun we have together!</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #ff1493; font-family: Comic Sans MS;'>
        <h4>💕 You make every moment special 💕</h4>
        <p>This app was created with love, just for us!</p>
    </div>
    """, unsafe_allow_html=True)

# Screenshot instructions
st.markdown("---")
st.markdown("### 📷 Taking Screenshots")
st.info("""
**To capture a photo:**
1. Right-click on the video feed
2. Select 'Save Image As...' or use your system's screenshot tool
3. Or use the screenshot button that appears when you hover over the video!

*Note: The video feed runs in your browser, so you can take screenshots anytime!*
""")
