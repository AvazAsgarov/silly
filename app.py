"""
Romantic Photo Filter App 💕
A fun and playful app with silly filters for you and your girlfriend!
Upload a photo and apply fun filters!
"""
import os
import sys

# Suppress TensorFlow / MediaPipe logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
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
    static_image_mode=True,
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


def apply_big_eyes(frame, landmarks, h, w, eye_scale):
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
                radius = int(np.linalg.norm(eye_points[0] - eye_center) * eye_scale)
                
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


def apply_funny_mouth(frame, landmarks, h, w, mouth_stretch):
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
                                     fx=mouth_stretch, fy=1.0,
                                     interpolation=cv2.INTER_LINEAR)
                
                sh, sw = stretched.shape[:2]
                start_x = max(0, (sw - (x2-x1)) // 2)
                crop_w = min(sw - start_x, x2 - x1)
                
                frame[y1:y2, x1:x1+crop_w] = stretched[:y2-y1, start_x:start_x+crop_w]
    except Exception as e:
        pass
    
    return frame


def apply_rainbow_hair(frame, landmarks, h, w):
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


def apply_cartoon_effect(frame):
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


def apply_pixelated_face(frame, landmarks, h, w, pixel_size):
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
                              (face_region.shape[1] // pixel_size, 
                               face_region.shape[0] // pixel_size),
                              interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (face_region.shape[1], face_region.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            
            frame[y1:y2, x1:x2] = pixelated
    except Exception as e:
        pass
    
    return frame


def apply_silly_colors(frame):
    """
    FILTER: Silly Colors (Inverted/Weird Colors)
    Inverts or shifts colors for a silly effect
    """
    # Convert to HSV and shift hue
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + 90) % 180  # Shift hue
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_floating_hearts(frame):
    """
    DECORATION: Floating Hearts
    Draws random hearts on the image
    """
    h, w = frame.shape[:2]
    
    # Generate random heart positions
    np.random.seed(42)  # For consistent hearts
    num_hearts = 15
    
    for i in range(num_hearts):
        x = np.random.randint(50, w - 50)
        y = np.random.randint(50, h - 50)
        
        # Draw heart
        heart_size = np.random.randint(15, 30)
        cv2.circle(frame, (x - heart_size//4, y), heart_size//2, (147, 20, 255), -1)
        cv2.circle(frame, (x + heart_size//4, y), heart_size//2, (147, 20, 255), -1)
        pts = np.array([[x, y + heart_size//2], 
                      [x - heart_size//2, y], 
                      [x + heart_size//2, y]], np.int32)
        cv2.fillPoly(frame, [pts], (147, 20, 255))
    
    return frame


def process_image(image, filters, eye_scale, mouth_stretch, pixel_size):
    """
    Main function to process the uploaded image with selected filters
    """
    # Convert PIL to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Process face landmarks
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Apply filters based on selection
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        if filters["big_eyes"]:
            img = apply_big_eyes(img, landmarks, h, w, eye_scale)
        
        if filters["funny_mouth"]:
            img = apply_funny_mouth(img, landmarks, h, w, mouth_stretch)
        
        if filters["rainbow_hair"]:
            img = apply_rainbow_hair(img, landmarks, h, w)
        
        if filters["pixelated"]:
            img = apply_pixelated_face(img, landmarks, h, w, pixel_size)
    
    # Apply filters that don't need face detection
    if filters["cartoon"]:
        img = apply_cartoon_effect(img)
    
    if filters["silly_colors"]:
        img = apply_silly_colors(img)
    
    # Add floating hearts decoration
    if filters["hearts"]:
        img = draw_floating_hearts(img)
    
    # Convert back to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


# Main app layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📸 Upload Your Photo")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a photo...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a photo to apply romantic filters!"
    )
    
    if uploaded_file is not None:
        # Load the image
        original_image = Image.open(uploaded_file)
        
        # Display original
        st.markdown("#### Original Photo")
        st.image(original_image, use_container_width=True)
        
        # Process button
        if st.button("✨ Apply Filters ✨", use_container_width=True):
            with st.spinner("Adding love and magic... 💕"):
                # Process the image
                processed_image = process_image(
                    original_image, 
                    filters, 
                    eye_scale, 
                    mouth_stretch, 
                    pixel_size
                )
                
                # Store in session state
                st.session_state['processed_image'] = processed_image
        
        # Display processed image if it exists
        if 'processed_image' in st.session_state:
            st.markdown("---")
            st.markdown("#### 💕 Filtered Photo 💕")
            st.image(st.session_state['processed_image'], use_container_width=True)
            
            # Download button
            img_pil = Image.fromarray(st.session_state['processed_image'])
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="💾 Download Filtered Photo",
                data=byte_im,
                file_name=f"love_filter_{timestamp}.png",
                mime="image/png",
                use_container_width=True
            )
    else:
        # Show placeholder
        st.info("👆 Upload a photo to get started!")
        st.image("https://via.placeholder.com/600x400/ffb6c1/ff1493?text=Upload+a+Photo+💕", 
                use_container_width=True)

with col2:
    st.markdown("### 💝 Love Notes")
    st.info("👉 Toggle filters in the sidebar!")
    st.success("📸 Adjust intensity sliders for more fun!")
    st.warning("💕 Hearts are enabled by default to make everything more romantic!")
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload your photo using the button
    2. Select filters from the sidebar
    3. Adjust intensity sliders
    4. Click "Apply Filters" button
    5. Download your filtered photo!
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
