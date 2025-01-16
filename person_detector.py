"""
This project uses the YOLOv8 object detection model from the Ultralytics library.

The Ultralytics library is licensed under the GNU General Public License v3.0 (GPL-3.0).
For more information, visit:
https://github.com/ultralytics/ultralytics
"""

from ultralytics import YOLO
import cv2
from config import PERSON_DETECTOR_MODEL_PATH

class PersonDetector:
    def __init__(self, model_path=PERSON_DETECTOR_MODEL_PATH, input_size=(640, 640), confidence_threshold=0.8):
        """
        Initializes the PersonDetector class.

        :param model_path: str
            Path to the YOLO model file.
        :param input_size: tuple
            Input size (width, height) for inference.
        :param confidence_threshold: float
            Minimum confidence threshold for detecting persons.
        """
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        Detects persons in the input frame using the YOLOv8 model.

        :param frame: numpy.ndarray
            Input image in which to detect persons.
        :return: list of tuples
            Detection results in the format [(bbox, confidence, class_id), ...], where:
                - bbox: list of int
                    Bounding box coordinates [x1, y1, x2, y2].
                - confidence: float
                    Confidence score of the detection.
                - class_id: int
                    Class ID of the detected object (0 for persons).
        """
        # Get original frame dimensions
        original_height, original_width = frame.shape[:2]

        # Resize the frame to the model's input size
        resized_frame = cv2.resize(frame, self.input_size)

        # Run inference with the YOLO model
        results = self.model(resized_frame, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes.data:
                # Extract bounding box, confidence, and class ID
                x1, y1, x2, y2, conf, cls = box.tolist()
                if int(cls) == 0 and conf >= self.confidence_threshold:  # Filter for persons only
                    # Scale bounding box coordinates back to original frame size
                    x1 = int(x1 * original_width / self.input_size[0])
                    y1 = int(y1 * original_height / self.input_size[1])
                    x2 = int(x2 * original_width / self.input_size[0])
                    y2 = int(y2 * original_height / self.input_size[1])

                    # Format bounding box and append to detections
                    bbox = [x1, y1, x2, y2]
                    detections.append((bbox, conf, 0))  # Class ID is 0 for persons

        return detections

