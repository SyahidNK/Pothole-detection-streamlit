# 🚧 Pothole Detection and Counting Web App

A web-based application for **detecting and counting road potholes** using **YOLO Deep Learning model** and **Streamlit**.

---

## 📌 Project Overview
This application is designed to automatically detect and count potholes on road surfaces from **images, videos, or live webcam input**.  
The system leverages a YOLO-based object detection model and provides an interactive web interface built with Streamlit.

Based on the detection results, bounding boxes are displayed along with the **total number of detected potholes**.

---

## 🚀 Features
- Multiple detection modes:
  - 🖼 Image
  - 🎥 Video
  - 📷 Webcam
- Adjustable **confidence threshold**
- Real-time pothole detection
- Automatic object counting
- Bounding box visualization
- Detection timestamp
- User-friendly Streamlit interface

---

## 🖥 User Interface Preview
Main components available in the web application:
- Detection mode selector (Image / Video / Webcam)
- Confidence threshold slider
- File uploader (drag & drop or browse)
- Detection result display
- Recent uploaded video history

---

## 🛠 Tech Stack
- Python
- Streamlit
- OpenCV
- NumPy
- Pillow (PIL)
- YOLO (Ultralytics)

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── ekspor_model.py # YOLO model export / helper script
├── utils/ # Utility and helper functions
├── README.md
├── requirements.txt
├── .gitignore

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
git clone https://github.com/SyahidNK/Pothole-detection-streamlit.git
cd Pothole-detection-streamlit

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py q