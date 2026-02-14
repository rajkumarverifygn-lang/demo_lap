import streamlit as st
from roboflow import Roboflow
from PIL import Image
import os
import cv2
import numpy as np

# --- 1. CONFIGURATION ---
# PASTE YOUR API KEY HERE (I removed it for your security)
API_KEY = "vJHcI10DU9vZTQvI24zE"

# UPDATED MODEL ID FROM YOUR SCREENSHOT
# Format: workspace / project / version
MODEL_ID = "annovfgn/lalala-ins/1"

LOGO_PATH = "logo.png"
CONFIDENCE = 5




# --- NEW: FILTER CLASSES HERE ---
# Only these classes will be drawn. Add/Remove names as needed.
ALLOWED_CLASSES = ["damage_bbox", "damage_s", "damage_seg", "scratch"] 

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="Laptop Defect Demo", layout="centered")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    h1 {text-align: center; color: #333;}
    </style>
""", unsafe_allow_html=True)

# --- 3. CUSTOM DRAWING FUNCTION ---
def draw_predictions(image, predictions):
    # Convert to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
    
    # Create overlay for transparency
    overlay = image_cv.copy()

    for pred in predictions:
        class_name = pred['class']
        
        # --- THE FILTER LOGIC ---
        # If the detected class is NOT in our allowed list, skip it.
        if class_name not in ALLOWED_CLASSES:
            continue
        
        # Extract data
        x, y = int(pred['x']), int(pred['y'])
        w, h = int(pred['width']), int(pred['height'])
        conf = int(pred['confidence'] * 100)
        
        # Calculate Corners
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # 1. DRAW POLYGON (Yellow Mask)
        if "points" in pred:
            pts = np.array([(p['x'], p['y']) for p in pred['points']], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], (0, 255, 255)) # Yellow
        
        # 2. DRAW BOX
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 3. DRAW LABEL
        label_text = f"{class_name} {conf}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        
        # Green Background for Text
        cv2.rectangle(image_cv, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 255, 255), -1)
        # Black Text
        cv2.putText(image_cv, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Blend overlay (transparency)
    alpha = 0.4 
    image_result = cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0)

    return cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

# --- 4. INITIALIZE MODEL ---
@st.cache_resource
def get_model():
    rf = Roboflow(api_key=API_KEY)
    project_version = MODEL_ID.split("/")
    project = rf.workspace(project_version[0]).project(project_version[1])
    model = project.version(project_version[2]).model
    return model

# --- 5. MAIN LOGIC ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

st.markdown("<h1>Demo for Laptop Defect</h1>", unsafe_allow_html=True)
st.write("---")

def run_app(image_source):
    temp_filename = "temp_source.jpg"
    image_source.save(temp_filename)

    with st.spinner("Analyzing..."):
        try:
            model = get_model()
            prediction = model.predict(temp_filename, confidence=CONFIDENCE)
            results_json = prediction.json()['predictions']
            
            # Filter the JSON list before counting
            filtered_results = [r for r in results_json if r['class'] in ALLOWED_CLASSES]
            
            if filtered_results:
                final_image = draw_predictions(image_source, results_json)
                st.image(final_image, caption="AI Analysis Result", use_container_width=True)
                st.success(f"✅ Found {len(filtered_results)} defects.")
            else:
                st.image(image_source, caption="No Allowed Defects Found", use_container_width=True)
                st.warning("No specific defects detected.")
            
        except Exception as e:
            st.error(f"Error: {e}")

option = st.radio("Choose Input Method:", ("Live Camera", "Upload Image"), horizontal=True)

if option == "Live Camera":
    st.subheader("Live Inspection")
    img_buffer = st.camera_input("Point camera at laptop surface")
    if img_buffer is not None:
        image = Image.open(img_buffer)
        run_app(image)

elif option == "Upload Image":
    st.subheader("Upload from Local Drive")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", width=300)
        if st.button("Detect Defects"):
            run_app(image)

