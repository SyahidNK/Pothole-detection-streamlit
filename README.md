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

Pothole-detection-streamlit/
│
├── app.py                      # Main Streamlit application
├── ekspor_model.py             # Model export / utility script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── models/                     # Trained YOLO model
│   └── best.pt
│   └── best.onnx
|
├── data/                       # Sample dataset (images)
│
├── result/                     # Detection output results
│
├── utils/                      # Helper functions
│
└── venv/                       # Virtual environment (not pushed to GitHub)

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/SyahidNK/Pothole-detection-streamlit.git
cd Pothole-detection-streamlit

2️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit App
streamlit run app.py