import cv2
from person_detector import PersonDetector
from lite_face_recognizer import LightweightFaceRecognition
from config import PERSON_DETECTOR_MODEL_PATH

def test_face_recognizer_video(video_path, person_model_path, face_model_dir):
    # モジュールの初期化
    person_detector = PersonDetector(model_path=person_model_path)
    face_recognizer = LightweightFaceRecognition(face_model_dir)

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
        person_detections = person_detector.detect(frame)

        for bbox, confidence, class_id in person_detections:
            x1, y1, x2, y2 = bbox
            person_roi = frame[y1:y2, x1:x2]

            # 顔検出
            face_box = face_recognizer.detect_faces(person_roi)
            if not face_box:
                print("No face detected in person ROI.")
                continue

            # 顔認証
            preprocessed_face = face_recognizer.preprocess_face(person_roi, face_box)
            embedding = face_recognizer.predict_embedding(preprocessed_face)

            # データベースと照合
            for person_id, embeddings in face_recognizer.database.items():
                similarities = [
                    cosine_similarity([embedding.flatten()], [db_embedding.flatten()])[0][0]
                    for db_embedding in embeddings
                ]
                max_similarity = max(similarities) if similarities else 0
                print(f"Detected {person_id} with similarity: {max_similarity:.2f}")

            # 検出結果の描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Similarity: {max_similarity:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 結果の表示
        cv2.imshow("Face Recognition - Video", frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# 実行
if __name__ == "__main__":
    test_video_path = "data\\VideoAfterEdit\\Front\\IMG_1945.MOV"  # テスト用動画のパスを指定
    test_person_model_path = PERSON_DETECTOR_MODEL_PATH  # YOLOv8モデルのパスを指定
    test_face_model_dir = "Bus_Occupant_Detect/mobileface_v1.0_infer"  # 顔認証モデルのパスを指定

    test_face_recognizer_video(test_video_path, test_person_model_path, test_face_model_dir)
