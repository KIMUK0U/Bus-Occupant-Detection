import cv2
from person_detector import PersonDetector

def main():
    # PersonDetectorの初期化
    detector = PersonDetector(model_path="yolov8n.pt")

    # カメラ映像の取得
    cap = cv2.VideoCapture(0)

    # カメラが正常に開けたかを確認
    if not cap.isOpened():
        print("Error: カメラが開けませんでした。")
        return

    # カメラの解像度を設定（必要に応じて変更）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: フレームの取得に失敗しました。")
                break

            # Person検出
            detections = detector.detect(frame)

            # 検出結果を描画
            for bbox, conf, class_id in detections:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 結果を表示
            cv2.imshow("Person Detection", frame)

            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("終了キーが押されました。")
                break
    finally:
        # リソース解放
        cap.release()
        cv2.destroyAllWindows()
        print("カメラとウィンドウを解放しました。")

if __name__ == "__main__":
    main()
