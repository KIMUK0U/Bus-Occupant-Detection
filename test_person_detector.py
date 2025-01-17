import cv2
from person_detector import PersonDetector
from config import PERSON_DETECTOR_MODEL_PATH

def test_person_detector_video(video_path, model_path="path/to/yolov8_model.pt"):
    # PersonDetectorの初期化
    detector = PersonDetector(model_path=model_path)

    # 動画キャプチャの初期化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 人物検出
        detections = detector.detect(frame)

        # 検出結果の描画
        for bbox, confidence, class_id in detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 動画フレームを表示
        cv2.imshow("Person Detection - Video", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

# 実行
if __name__ == "__main__":
    test_video_path = "data\VideoAfterEdit\Front\E_beside9.MOV"  # テスト用動画のパスを指定
    test_model_path = PERSON_DETECTOR_MODEL_PATH  # YOLOv8モデルのパスを指定
    test_person_detector_video(test_video_path, test_model_path)
