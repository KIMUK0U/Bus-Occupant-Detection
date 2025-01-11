import cv2
from person_detector import PersonDetector
from face_recognizer import FaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity

"""
顔認証までのアルゴリズムがプログラムされた関数群
"""
def initialize_modules():
    """モジュールの初期化を行う"""
    person_detector = PersonDetector()
    # face_detector = FaceDetector() 使用する必要ない
    face_recognizer = FaceRecognition()
    tracker = DeepSort()

    # ユーザー登録
    # 登録データの辞書
    persons_data = {
        "KIMURA KOTA": [
            "data/faces/user1/face_0.jpg",
            "data/faces/user1/face_1.jpg",
            "data/faces/user1/face_2.jpg",
            "data/faces/user1/face_3.jpg",
            "data/faces/user1/face_4.jpg",
            "data/faces/user1/face_5.jpg"
        ],
        "KOIKE YUTA": [
            "data/faces/user2/face_0.jpg",
            "data/faces/user1/face_1.jpg"
        ],
        "YASUI": [
            "data/faces/user3/face_0.jpg"
        ],
        "KUROZAWA": [
            "data/faces/user4/face_0.jpg",
            "data/faces/user4/face_1.jpg"
        ],
        "HOUJOU": [
            "data/faces/user1/face_0.jpg"
        ]
    }
    # 一括登録
    face_recognizer.enroll_multiple_persons(persons_data)

    return person_detector, face_recognizer, tracker

def detect_and_recognize_faces(person_roi, face_recognizer, track_id, tracked_names):
    """人物領域から顔認証を行う"""
    if track_id in tracked_names and tracked_names[track_id] != "unknown":
        # すでに名前がわかっている場合、顔認識をスキップして再度認証しなくてもいい
        return tracked_names[track_id], 0.0

    faces = face_recognizer.app.get(person_roi)
    name = "unknown"
    max_similarity_global = 0.0  # ここで類似度を初期化

    if len(faces) > 0:
        print(f"Detected {len(faces)} face(s) in person ROI")
        for face in faces:
            input_embedding = face.normed_embedding

            # データベースと照合
            max_similarity_local = 0
            matched_person = "unknown"
            for person_id, embeddings in face_recognizer.database.items():
                similarities = [cosine_similarity([input_embedding], [embedding])[0][0] for embedding in embeddings]
                similarity = max(similarities) if similarities else 0

                if similarity > max_similarity_local:
                    max_similarity_local = similarity
                    matched_person = person_id

            # 最も高い類似度を持つ person_id と比較
            if max_similarity_local > max_similarity_global:
                max_similarity_global = max_similarity_local
                name = matched_person

        # 類似度が閾値を超えた場合に認証成功とする
        if max_similarity_global > 0.5:
            tracked_names[track_id] = name
            print(f"Matched with {name}, similarity: {max_similarity_global:.2f}")
            return name, max_similarity_global
        else:
            print(f"No matched. max_similarity: {max_similarity_global:.2f}")
            return "unknown", max_similarity_global

    print("No match found or no faces detected.")
    return name, max_similarity_global



def process_frame(frame, person_detector, face_recognizer, tracker, tracked_names):
    person_detections = person_detector.detect(frame)
    formatted_detections = [
        (bbox, confidence, class_id) for bbox, confidence, class_id in person_detections
    ]
    tracks = tracker.update_tracks(formatted_detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()

        person_roi = frame[int(ltrb[1]):int(ltrb[3]), int(ltrb[0]):int(ltrb[2])]
        if person_roi.size == 0:
            print("Error: person_roi is empty or invalid!")
            name = "Unknown"
            similarity = 0.0
        else:
            name, similarity = detect_and_recognize_faces(person_roi, face_recognizer, track_id, tracked_names)

        draw_results(frame, ltrb, track_id, name, similarity)


def draw_results(frame, ltrb, track_id, name, similarity):
    """描画を行う"""
    # 人物矩形の描画
    cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {track_id}, Name: {name}",
                (int(ltrb[0]), int(ltrb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ---【追加】 類似度を右下に描画する ----------------
    # 画面サイズを取得
    h, w = frame.shape[:2]
    # 類似度を小数点以下2桁で丸めたテキスト
    similarity_text = f"Similarity: {similarity:.2f}"

    # 右下に描画する座標(例: 下マージン10px, 右マージン200pxあたり)
    text_x = w - 200
    text_y = h - 10
    cv2.putText(frame, similarity_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # -------------------------------------------------

def main():
    person_detector, face_recognizer, tracker = initialize_modules()
    tracked_names = {}

    count = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        count+=1
        if count%5 == 0:# 5フレームごとに検出
            
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame, person_detector, face_recognizer, tracker, tracked_names)

            cv2.imshow("Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            continue
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
