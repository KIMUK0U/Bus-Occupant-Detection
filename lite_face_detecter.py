from mtcnn.mtcnn import MTCNN
class FaceDetector:
    def __init__(self, confidence_threshold=0.4):
        """
        MTCNNの初期化と信頼度の閾値を設定
        """
        self.detector = MTCNN()  # MTCNNの初期化
        self.confidence_threshold = confidence_threshold  # 信頼度の閾値

    def detect_faces(self, image):
        """
        顔検出を行い、信頼度スコアでフィルタリング
        """
        detections = self.detector.detect_faces(image)
        if len(detections) == 0:
            return None

        # 信頼度スコアでフィルタリング
        filtered_faces = [d for d in detections if d['confidence'] >= self.confidence_threshold]

        if not filtered_faces:
            return None

        # フィルタリング後の顔ボックスを返す
        return [face['box'] for face in filtered_faces]
