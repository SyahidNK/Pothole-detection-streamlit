from ultralytics import YOLO

# Load model .pt
model = YOLO("models/best.pt")

# Ekspor ke ONNX dengan post-processing (NMS)
model.export(format="onnx", dynamic=False, simplify=True, half=False,imgsz=640, conf=0.4, iou=0.5, nms=True)