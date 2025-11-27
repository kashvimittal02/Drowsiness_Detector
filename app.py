import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import streamlit.components.v1 as components
import time
import os
import base64

# ----------------------------
# Config / defaults
# ----------------------------
DEFAULT_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
DEFAULT_ALARM_PATH = r"C:\Users\hp\OneDrive\Desktop\Real-Time-Drowsiness-Detection-System\Alert.wav"

# ----------------------------
# Utility functions
# ----------------------------
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    if D == 0:
        return 0.0
    return (A + B + C) / (3.0 * D)

def make_autoplay_html_from_b64(b64_audio, mime="audio/wav"):
    """
    Create HTML with aggressive autoplay attempts using multiple methods.
    Uses a unique ID per call to force browser to create new audio element.
    """
    unique_id = int(time.time() * 1000000)  # microsecond precision for uniqueness
    html = f"""
    <audio id="alarm_{unique_id}" style="display:none;">
        <source src="data:{mime};base64,{b64_audio}" type="{mime}">
    </audio>
    <script>
    (function() {{
        const audio = document.getElementById('alarm_{unique_id}');
        
        // Method 1: Direct play with promise handling
        audio.volume = 1.0;
        audio.currentTime = 0;
        const playPromise = audio.play();
        
        if (playPromise !== undefined) {{
            playPromise.catch(err => {{
                console.log('Play failed, trying Web Audio API:', err);
                playWithWebAudio();
            }});
        }}
        
        // Method 2: Web Audio API fallback
        function playWithWebAudio() {{
            try {{
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                const audioCtx = new AudioContext();
                
                // Decode base64
                const binaryString = atob('{b64_audio}');
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                
                audioCtx.decodeAudioData(bytes.buffer).then(buffer => {{
                    const source = audioCtx.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioCtx.destination);
                    
                    if (audioCtx.state === 'suspended') {{
                        audioCtx.resume().then(() => source.start(0));
                    }} else {{
                        source.start(0);
                    }}
                }});
            }} catch(e) {{
                console.log('Web Audio failed:', e);
            }}
        }}
    }})();
    </script>
    """
    return html

def read_file_as_b64_bytes(path_or_bytes, is_bytes=False):
    """
    Accepts either a path (str) or raw bytes (if is_bytes=True).
    Returns (b64string, mime) or (None, None) on failure.
    """
    try:
        if is_bytes:
            data = path_or_bytes
            mime = "audio/wav"
        else:
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            ext = os.path.splitext(path_or_bytes)[1].lower()
            if ext in [".mp3"]:
                mime = "audio/mpeg"
            elif ext in [".ogg"]:
                mime = "audio/ogg"
            else:
                mime = "audio/wav"
        b64 = base64.b64encode(data).decode("ascii")
        return b64, mime
    except Exception as e:
        return None, None

# ----------------------------
# UI
# ----------------------------
st.set_page_config(layout="wide", page_title="Drowsiness + Yawn Detection")
st.title("😴 Drowsiness + Yawn Detection — autoplay-ready alarm")

col1, col2 = st.columns([3,1])

with col2:
    EAR_THRESHOLD = st.slider("EAR threshold", 0.15, 0.40, 0.25, 0.01)
    EAR_CONSEC = st.slider("EAR consecutive frames", 1, 40, 15, 1)
    ENABLE_YAWN = st.checkbox("Enable Yawn Detection", value=True)
    MAR_THRESHOLD = st.slider("MAR threshold", 0.25, 1.0, 0.55, 0.01)
    MAR_CONSEC = st.slider("MAR consecutive frames", 1, 40, 12, 1)
    st.write("---")
    predictor_path = st.text_input("Predictor file path (68 landmarks)", DEFAULT_PREDICTOR)

with col1:
    run = st.checkbox("Start Webcam (click once to grant camera permission)")
    FRAME_WINDOW = st.image([])

# ----------------------------
# Session state
# ----------------------------
if "ear_counter" not in st.session_state:
    st.session_state.ear_counter = 0
if "mar_counter" not in st.session_state:
    st.session_state.mar_counter = 0
if "last_alarm_ts" not in st.session_state:
    st.session_state.last_alarm_ts = 0.0

# ----------------------------
# Validate predictor & prepare alarm b64
# ----------------------------
if not os.path.exists(predictor_path):
    st.error(f"Missing predictor file: {predictor_path}. Place '{predictor_path}' in the app folder or change the path.")
    st.stop()

# Prepare b64 alarm from DEFAULT_ALARM_PATH
alarm_b64 = None
alarm_mime = None
if DEFAULT_ALARM_PATH and os.path.exists(DEFAULT_ALARM_PATH):
    alarm_b64, alarm_mime = read_file_as_b64_bytes(DEFAULT_ALARM_PATH)
    st.success(f"✅ Alarm loaded: {DEFAULT_ALARM_PATH}")
else:
    st.error(f"❌ Alarm file not found: {DEFAULT_ALARM_PATH}. Please check the path.")

# ----------------------------
# Initialize dlib models
# ----------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Placeholder for alarm audio (reused for each alarm trigger)
alarm_container = st.empty()

# ----------------------------
# Run loop
# ----------------------------
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure camera is free and permissions granted.")
    else:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame read failed.")
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray, 0)

                drowsy = False
                yawned = False

                for face in faces:
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)

                    leftEye = shape[36:42]
                    rightEye = shape[42:48]
                    mouth = shape[48:68]

                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    mar = mouth_aspect_ratio(mouth) if ENABLE_YAWN else 0.0

                    cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
                    cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)
                    cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255,0,0), 1)

                    if ear < EAR_THRESHOLD:
                        st.session_state.ear_counter += 1
                        if st.session_state.ear_counter >= EAR_CONSEC:
                            drowsy = True
                    else:
                        st.session_state.ear_counter = 0

                    if ENABLE_YAWN:
                        if mar > MAR_THRESHOLD:
                            st.session_state.mar_counter += 1
                            if st.session_state.mar_counter >= MAR_CONSEC:
                                yawned = True
                        else:
                            st.session_state.mar_counter = 0

                    cv2.putText(frame, f"EAR: {ear:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                    if ENABLE_YAWN:
                        cv2.putText(frame, f"MAR: {mar:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 2)

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                alarm_trigger = drowsy or yawned
                now = time.time()
                if alarm_trigger and (now - st.session_state.last_alarm_ts > 3.0):
                    st.session_state.last_alarm_ts = now
                    if drowsy:
                        st.warning("🚨 DROWSINESS DETECTED!")
                    if yawned:
                        st.info("😮 YAWN DETECTED!")

                    # Play alarm when drowsiness or yawn detected
                    if alarm_b64:
                        try:
                            html = make_autoplay_html_from_b64(alarm_b64, mime=alarm_mime or "audio/wav")
                            with alarm_container:
                                components.html(html, height=0)
                        except Exception as e:
                            st.error(f"Unable to play alarm: {e}")
                    else:
                        st.error("No alarm file available to play. Upload one in the uploader.")

                time.sleep(0.03)
        except Exception as e:
            st.error(f"Error in webcam loop: {e}")
        finally:
            cap.release()
else:
    st.info("Tick 'Start Webcam' to begin. (Clicking once grants camera permission and helps autoplay.)")