import cv2
import time
from face_detector import FaceDetector
from person_detector import PersonDetector

def main():
    # PersonDetector と FaceDetector を初期化
    person_detector = PersonDetector()
    face_detector = FaceDetector()

    # カメラの初期化（デバイスID 0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # FPS計測用
    prev_time = time.time()
    frame_count = 0
    frame_id = 0
    fps = 0.0  # 初期値を設定

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame_id += 1
        if frame_id % 2 != 0:
            continue

        # 描画用フレームを大きくリサイズ
        original_height, original_width = frame.shape[:2]
        display_frame = cv2.resize(frame, (original_width * 2, original_height * 2))

        # Person detection
        person_detections = person_detector.detect(frame)
        for bbox, confidence, class_id in person_detections:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1 * 2, y1 * 2), (x2 * 2, y2 * 2), (0, 255, 0), 2)
            cv2.putText(
                display_frame, f"Person: {confidence:.2f}", (x1 * 2, y1 * 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # 人の領域を切り出して顔検出を行う
            person_roi = frame[y1:y2, x1:x2]
            face_detections = face_detector.detect(person_roi)
            for fx1, fy1, fx2, fy2, fconfidence in face_detections:
                # 元のフレーム座標にスケールを戻す
                face_x1 = x1 + fx1
                face_y1 = y1 + fy1
                face_x2 = x1 + fx2
                face_y2 = y1 + fy2
                cv2.rectangle(display_frame, (face_x1 * 2, face_y1 * 2), (face_x2 * 2, face_y2 * 2), (255, 0, 0), 2)
                cv2.putText(
                    display_frame, f"Face: {fconfidence:.2f}", (face_x1 * 2, face_y1 * 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                )

        # FPS計測
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - prev_time

        if elapsed_time >= 1.0:  # 1秒ごとにFPSを更新
            fps = frame_count / elapsed_time
            prev_time = current_time
            frame_count = 0
            print(f"FPS: {fps:.2f}")

        # FPSをフレームに描画
        cv2.putText(
            display_frame, f"FPS: {fps:.2f}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # フレームを表示
        cv2.imshow("Person and Face Detection Demo", display_frame)

        # 'q'キーが押されたらループを終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()