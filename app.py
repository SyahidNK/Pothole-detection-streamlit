import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
from datetime import datetime
import time
import base64

def render_looping_video_html(path):
    with open(path, "rb") as f:
        video_bytes = f.read()
        encoded = base64.b64encode(video_bytes).decode()

        video_html = f"""
        <video autoplay loop muted controls width="100%" style="border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        """
        st.markdown(video_html, unsafe_allow_html=True)



# Import detector AFTER streamlit to avoid conflicts
from utils.detector import YOLODetector

# Setup halaman
st.set_page_config(page_title="Pothole Detection App", layout="centered")
st.title("Deteksi dan Counting Objek Lubang Jalan")

# Create results directory if it doesn't exist
os.makedirs("result/video", exist_ok=True)

# Sidebar untuk mode dan parameter
mode = st.sidebar.radio("Pilih Mode Deteksi:", ["Gambar", "Video", "Webcam"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# Inisialisasi model dengan caching untuk performa
@st.cache_resource
def load_model(model_path, conf_threshold):
    """Load model with caching to avoid reloading"""
    return YOLODetector(model_path, conf_threshold)

# Load model
model_path = "weights/best.onnx"
detector = load_model(model_path, conf_threshold)

def get_video_writer(input_path, output_path, cap):
    """Create video writer with same properties as input video"""
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:  # Handle invalid fps
        fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return out

def generate_output_filename(original_filename):
    """Generate output filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(original_filename)
    return f"{name}_detected_{timestamp}.mp4"

# Mode Gambar
if mode == "Gambar":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Load and display original image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        
        # Create columns for before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_np, caption="Original Image", use_container_width=True)
        
        with col2:
            with st.spinner("Detecting potholes..."):
                # Detect potholes
                result, count, detections = detector.detect_with_count(image_np)
            
            st.image(result, caption=f"Detection Result ({count} potholes found)", 
                    channels="BGR", use_container_width=True)
        
        # Show detection details
        if count > 0:
            st.success(f"🎯 Found {count} pothole(s)!")
            with st.expander("Detection Details"):
                for i, det in enumerate(detections):
                    st.write(f"**Pothole {i+1}:** Confidence = {det['score']:.3f}")
        else:
            st.info("No potholes detected in this image.")

# Mode Video
elif mode == "Video":
    loop_result = st.sidebar.checkbox("🔁 Putar Ulang Hasil Deteksi", value=False)
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        # Generate output filename
        output_filename = generate_output_filename(uploaded_video.name)
        output_path = os.path.join("result/video", output_filename)
        
        # Save uploaded video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        # Process video
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not open video file")
        else:
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            st.info(f"📹 Video Info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
            
            # Create video writer
            out = get_video_writer(tfile.name, output_path, cap)
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            stframe = st.empty()
            
            frame_count = 0
            total_detections = 0
            
            # Process each frame
            start_time = time.time()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect potholes
                result, count, _ = detector.detect_with_count(frame)
                total_detections += count
                
                # Write frame to output video
                out.write(result)
                
                # Update display every 10 frames for better performance
                if frame_count % 10 == 0:
                    stframe.image(result, channels="BGR", 
                                caption=f"Processing frame {frame_count + 1}/{total_frames}")
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                
                elapsed_time = time.time() - start_time
                fps_actual = frame_count / elapsed_time if elapsed_time > 0 else 0
                status_text.text(f"Processing: {frame_count}/{total_frames} frames | "
                               f"Speed: {fps_actual:.1f} FPS | Detections: {total_detections}")
            
            # Clean up
            cap.release()
            out.release()
            os.remove(tfile.name)
            
            # Show completion
            progress_bar.progress(1.0)
            processing_time = time.time() - start_time
            status_text.text(f"✅ Processing complete! Time: {processing_time:.1f}s")
            
            st.success(f"🎉 Video processed successfully!")
            st.info(f"📁 Output saved to: `{output_path}`")
            st.metric("Total Potholes Detected", total_detections)
            
            # Show file info and download
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                st.write(f"📊 Output file size: {file_size:.2f} MB")
                
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=file.read(),
                        file_name=output_filename,
                        mime="video/mp4"
                    )
                # 🔁 Tampilkan hasil video jika loop diaktifkan
                if loop_result:
                    st.info("🔁 Memutar ulang video hasil deteksi...")
                    with st.spinner("🔁 Memuat ulang video hasil deteksi..."):
                        render_looping_video_html(output_path)
# Mode Webcam
elif mode == "Webcam":
    st.warning("📷 Click the button below to start webcam detection.")
    
    # Webcam options
    col1, col2 = st.columns(2)
    with col1:
        record_webcam = st.checkbox("🔴 Record webcam output", value=False)
    with col2:
        if record_webcam:
            duration = st.number_input("Recording duration (seconds)", 
                                     min_value=5, max_value=300, value=30)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_webcam = st.button("▶️ Start Detection", type="primary")
    with col2:
        if 'webcam_running' in st.session_state and st.session_state.webcam_running:
            stop_webcam = st.button("⏹️ Stop Detection")
        else:
            stop_webcam = False
    
    if start_webcam:
        st.session_state.webcam_running = True
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Error: Could not access webcam")
            st.session_state.webcam_running = False
        else:
            stframe = st.empty()
            stats_placeholder = st.empty()
            
            # Setup recording if enabled
            out = None
            if record_webcam:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                webcam_output_path = f"result/video/webcam_detected_{timestamp}.mp4"
                
                fps = 20
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(webcam_output_path, fourcc, fps, (width, height))
                
                st.info(f"🔴 Recording started - Duration: {duration} seconds")
                start_time = time.time()
            
            frame_count = 0
            total_detections = 0
            
            try:
                while st.session_state.get('webcam_running', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to read from webcam")
                        break
                    
                    # Detect potholes
                    result, count, _ = detector.detect_with_count(frame)
                    total_detections += count
                    
                    # Save frame if recording
                    if out is not None:
                        out.write(result)
                        
                        # Check recording duration
                        elapsed_time = time.time() - start_time
                        if elapsed_time >= duration:
                            st.success(f"✅ Recording completed! Saved to: `{webcam_output_path}`")
                            break
                    
                    # Display frame
                    stframe.image(result, channels="BGR", 
                                caption=f"Webcam Feed - Frame {frame_count}")
                    
                    # Update stats
                    stats_placeholder.metric("Total Detections", total_detections)
                    
                    frame_count += 1
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                st.error(f"Error during webcam processing: {e}")
            finally:
                cap.release()
                if out is not None:
                    out.release()
                    if os.path.exists(webcam_output_path):
                        file_size = os.path.getsize(webcam_output_path) / (1024 * 1024)
                        st.write(f"📊 Recorded file size: {file_size:.2f} MB")
                
                st.session_state.webcam_running = False

# Sidebar: Show saved videos
with st.sidebar:
    st.write("### 📁 Recent Videos")
    video_dir = "result/video"
    if os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov'))]
        if video_files:
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(os.path.join(video_dir, x)), 
                           reverse=True)
            
            for video_file in video_files[:5]:  # Show last 5 videos
                file_path = os.path.join(video_dir, video_file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                st.write(f"📄 **{video_file[:30]}{'...' if len(video_file) > 30 else ''}**")
                st.write(f"   📊 {file_size:.1f} MB")
                st.write(f"   🕒 {mod_time.strftime('%Y-%m-%d %H:%M')}")
                st.write("---")
        else:
            st.write("No saved videos yet")
    else:
        st.write("No videos directory found")

# Footer
st.markdown("---")
st.markdown("🚗 **Pothole Detection App** - Powered by YOLO & Streamlit")