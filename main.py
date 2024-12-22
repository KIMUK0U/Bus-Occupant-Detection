import cv2
from person_detector import PersonDetector
from face_detector import FaceDetector
from face_recognizer import FaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORTをインポート


def main():
    # モジュールの初期化
    person_detector = PersonDetector()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognition()

    # ユーザー登録
    face_recognizer.enroll_person(
        "User1", 
        [
            "data/faces/user1/face_0.jpg", 
            "data/faces/user1/face_1.jpg"
        ]
    )

    tracker = DeepSort()

    # カメラ映像を読み込む
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Person検出
        person_detections = person_detector.detect(frame)

        # DeepSort用のフォーマット変換
        formatted_detections = []
        for detection in person_detections:
            bbox, confidence, class_id = detection
            formatted_detections.append((bbox, confidence, class_id))

        # Step 2: トラッキング
        tracks = tracker.update_tracks(formatted_detections, frame=frame)

        # Step 3: 顔検出と認識
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            person_roi = frame[int(ltrb[1]):int(ltrb[3]), int(ltrb[0]):int(ltrb[2])]
            if person_roi is None or person_roi.size == 0:
                print("Error: person_roi is empty or invalid!")
                continue

            faces = face_detector.detect(person_roi)
            #デバック: 検出結果を確認
            if faces:
                print(f"Detected {len(faces)} face(s) in person ROI")
            else:
                print("No face detected in person ROI")
            for x1, y1, x2, y2, confidence in faces:
                # person_roiのサイズを取得
                h, w, _ = person_roi.shape

                # 座標の範囲を調整
                margin = 100  # 200ピクセルのマージンを追加
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                face_roi = person_roi[y1:y2, x1:x2]

                if face_roi is None or face_roi.size == 0:
                    print("Error: face_roi is empty!")
                    continue


                # 顔認証
                name = "unknown"
                for person_id in face_recognizer.database.keys():
                    result = face_recognizer.verify_person(face_roi, person_id)
                    #デバック:　認証結果を確認
                    print(f"Verification result for {person_id}: {result}")
                    if "Match" in result:
                        name = person_id
                        break

                x1 = max(0, x1 + margin)
                y1 = max(0, y1 + margin)
                x2 = min(w, x2 - margin)
                y2 = min(h, y2 - margin)
                # 結果を描画
                cv2.rectangle(frame, (int(ltrb[0]) + x1, int(ltrb[1]) + y1),
                              (int(ltrb[0]) + x2, int(ltrb[1]) + y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}, Name: {name}",
                            (int(ltrb[0]), int(ltrb[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 結果を表示
        cv2.imshow("Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
