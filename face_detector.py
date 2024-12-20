import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bboxC.xmin * w)
                y1 = int(bboxC.ymin * h)
                x2 = int((bboxC.xmin + bboxC.width) * w)
                y2 = int((bboxC.ymin + bboxC.height) * h)
                confidence = detection.score[0]
                faces.append((x1, y1, x2, y2, confidence))
        return faces
