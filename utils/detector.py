import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.4, ):
        """
        Initialize YOLO detector with PyTorch model (.pt file)
        
        Args:
            model_path: Path to .pt model file
            conf_threshold: Confidence threshold for detections
        """
        try:
            self.model = YOLO(model_path)  # Load the YOLO model
            self.conf_threshold = conf_threshold
            print(f"PyTorch model loaded successfully from: {model_path}")
            
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
            raise

    def detect(self, image):
        """
        Main detection method - maintains compatibility with existing app.py
        
        Args:
            image: Input image (BGR format for OpenCV or RGB for PIL)
            
        Returns:
            Image with drawn detections (BGR format for Streamlit)
        """
        try:
            # Convert BGR to RGB if needed (OpenCV uses BGR, YOLO expects RGB)
            if isinstance(image, np.ndarray):
                # Assume BGR format from OpenCV
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Already RGB format
                image_rgb = image
            
            # Run inference with confidence threshold
            results = self.model(image_rgb, conf=self.conf_threshold, verbose=False)
            
            # Get the annotated image
            annotated_image = results[0].plot()
            
            # Convert back to BGR for Streamlit display
            if isinstance(image, np.ndarray):
                return cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                return annotated_image
                
        except Exception as e:
            print(f"Detection error: {e}")
            # Return original image if detection fails
            return image

    def detect_with_count(self, image):
        """
        Detection method that also returns count information
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (result_image, detection_count, detections_info)
        """
        try:
            # Convert BGR to RGB if needed
            if isinstance(image, np.ndarray):
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run inference
            results = self.model(image_rgb, conf=self.conf_threshold, verbose=False)
            
            # Get results
            result = results[0]
            annotated_image = result.plot()
            
            # Extract detection information
            detections_info = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
                scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
                classes = result.boxes.cls.cpu().numpy()  # Get class IDs
                
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box
                    detections_info.append({
                        "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # Convert to xywh
                        "score": float(score),
                        "class_id": int(cls)
                    })
            
            # Convert back to BGR for Streamlit
            if isinstance(image, np.ndarray):
                result_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                result_image = annotated_image
            
            return result_image, len(detections_info), detections_info
            
        except Exception as e:
            print(f"Detection error: {e}")
            return image, 0, []

    def get_model_info(self):
        """Get model information"""
        return {
            "model_type": "PyTorch YOLO",
            "classes": self.model.names,
            "confidence_threshold": self.conf_threshold
        }