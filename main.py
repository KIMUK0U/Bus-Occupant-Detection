import cv2
from person_detector import PersonDetector
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORTをインポート


def main():
    # モジュールの初期化
    person_detector = PersonDetector()
    face_detector = FaceDetector()
    """
    face_recognizer = FaceRecognizer(
        known_face_encodings=[...],  # 登録済みユーザーの特徴量
        known_face_names=["User1", "User2", "User3"]
    )
    """
    tracker = DeepSort()

    # カメラ映像を読み込む
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Person検出
        person_detections = person_detector.detect(frame)

        # デバッグ: 検出結果を確認
        print(f"person_detections: {person_detections}")

        # DeepSort用のフォーマット変換
        formatted_detections = []
        for detection in person_detections:
            bbox, confidence, class_id = detection
            x1, y1, x2, y2 = bbox  # bboxを分解
            formatted_detections.append((bbox, confidence, class_id))  # 再フォーマット（この場合不要かもしれません）

        # デバッグ用
        print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, confidence: {confidence}")

        # Step 3: トラッキング
        tracks = tracker.update_tracks(formatted_detections, frame=frame)

        # Step 4: 顔検出と認識
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            person_roi = frame[int(ltrb[1]):int(ltrb[3]), int(ltrb[0]):int(ltrb[2])]

            faces = face_detector.detect(person_roi)
            for x1, y1, x2, y2, confidence in faces:
                """
                face_roi = person_roi[y1:y2, x1:x2]
                face_encoding = extract_face_encoding(face_roi)  # 顔の特徴量を取得する関数
                name = face_recognizer.recognize(face_encoding)
                """
                name = "unknown"

                # 結果を描画
                cv2.rectangle(frame, (int(ltrb[0]) + x1, int(ltrb[1]) + y1),
                              (int(ltrb[0]) + x2, int(ltrb[1]) + y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}, Name: {name}",
                            (int(ltrb[0]), int(ltrb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 結果を表示
        cv2.imshow("YOLO + Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
