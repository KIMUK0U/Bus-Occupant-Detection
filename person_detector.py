from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", input_size=(640, 640), confidence_threshold=0.8):
        """
        PersonDetectorクラスの初期化
        :param model_path: YOLOモデルのパス
        :param input_size: 推論時の入力サイズ（幅, 高さ）
        :param confidence_threshold: 検出の信頼度閾値
        """
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """
        入力フレームからpersonクラスを検出
        :param frame: 入力画像（numpy配列）
        :return: 検出結果 [(bbox, confidence, class_id), ...]
        """
        # フレームをリサイズ（必要に応じて）
        original_height, original_width = frame.shape[:2]
        resized_frame = cv2.resize(frame, self.input_size)

        # 推論を実行
        results = self.model(resized_frame, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes.data:
                # ボックス座標とクラスを取得
                x1, y1, x2, y2, conf, cls = box.tolist()
                if int(cls) == 0 and conf >= self.confidence_threshold:  # personクラスのみ
                    # ボックス座標を元の解像度にスケーリング
                    x1 = int(x1 * original_width / self.input_size[0])
                    y1 = int(y1 * original_height / self.input_size[1])
                    x2 = int(x2 * original_width / self.input_size[0])
                    y2 = int(y2 * original_height / self.input_size[1])

                    # DeepSort用のフォーマットに変換
                    bbox = [x1, y1, x2, y2]
                    detections.append((bbox, conf, 0))  # class_id を 0 に設定

        return detections
