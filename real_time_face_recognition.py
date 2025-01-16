import cv2
from person_detector import PersonDetector
from face_recognizer import FaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from attendance_manager import MinimalAttendanceManager

"""
顔認証までのアルゴリズムがプログラムされた関数
"""
def initialize_modules():
    """
    必要なモジュールを初期化し、顔認識データベースを構築する。

    :return: tuple
        (PersonDetector オブジェクト, FaceRecognition オブジェクト, DeepSort オブジェクト, MinimalAttendanceManager オブジェクト)
    """
    person_detector = PersonDetector()
    face_recognizer = FaceRecognition()
    tracker = DeepSort()
    attendance_manager = MinimalAttendanceManager()

    # ユーザー登録
    # 登録データの辞書
    persons_data = {
        "KIMURA KOTA": [
            "Bus_Occupant_Detect/faces/user1/face_0.jpg",
            "Bus_Occupant_Detect/faces/user1/face_1.jpg",
            "Bus_Occupant_Detect/faces/user1/face_2.jpg",
            "Bus_Occupant_Detect/faces/user1/face_3.jpg",
            "Bus_Occupant_Detect/faces/user1/face_4.jpg",
            "Bus_Occupant_Detect/faces/user1/face_5.jpg"
        ],
        "KOIKE YUTA": [
            "Bus_Occupant_Detect/faces/user2/face_0.jpg",
            "Bus_Occupant_Detect/faces/user1/face_1.jpg"
        ],
        "YASUI": [
            "Bus_Occupant_Detect/faces/user3/face_0.jpg",
            "Bus_Occupant_Detect/faces/user3/face_1.jpg",
            "Bus_Occupant_Detect/faces/user3/face_2.jpg",
            "Bus_Occupant_Detect/faces/user3/face_3.jpg"
        ],
        "KUROZAWA": [
            "Bus_Occupant_Detect/faces/user4/face_0.jpg",
            "Bus_Occupant_Detect/faces/user4/face_1.jpg"
        ],
        "HOUJOU": [
            "Bus_Occupant_Detect/faces/user1/face_0.jpg"
        ]
    }
    # 一括登録
    face_recognizer.enroll_multiple_persons(persons_data)
    # 登録データをデータベースに保存
    attendance_manager.create_user_table()  # テーブルの作成
    user_data = {f"user_{i+1}": name for i, name in enumerate(persons_data.keys())}
    attendance_manager.register_users(user_data)  # UserNameをテーブルに保存

    return person_detector, face_recognizer, tracker, attendance_manager

def detect_and_recognize_faces(person_roi, face_recognizer, track_id, tracked_names):
    """
    指定された人物領域から顔認証を実行し、登録データベースと照合する。

    :param person_roi: numpy.ndarray
        人物領域の画像。
    :param face_recognizer: FaceRecognition
        顔認識モジュール。
    :param track_id: int
        トラッキングID。
    :param tracked_names: dict
        トラッキング中の名前。
    :return: tuple
        認識された名前 (str) と類似度 (float)。
    """
    if track_id in tracked_names and tracked_names[track_id] != "unknown":
        return tracked_names[track_id], 0.0

    faces = face_recognizer.app.get(person_roi)  # 顔認証の関数
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
        if max_similarity_global > 0.6:
            tracked_names[track_id] = name
            print(f"Matched with {name}, similarity: {max_similarity_global:.2f}")
            return name, max_similarity_global
        else:
            print(f"No matched. max_similarity: {max_similarity_global:.2f}")
            return "unknown", max_similarity_global

    print("No match found or no faces detected.")
    return name, max_similarity_global

def process_frame(frame, person_detector, face_recognizer, tracker, attendance_manager, tracked_names, camera_id):
    """
    フレーム内の人物を検出し、顔認識を行い、入退室状態を更新する。

    :param frame: numpy.ndarray
        処理対象の画像フレーム。
    :param person_detector: PersonDetector
        人物検出モジュール。
    :param face_recognizer: FaceRecognition
        顔認識モジュール。
    :param tracker: DeepSort
        トラッキングモジュール。
    :param attendance_manager: MinimalAttendanceManager
        入退室管理モジュール。
    :param tracked_names: dict
        トラッキング中の名前。
    :param camera_id: int
        カメラID (0: 入室, 1: 退室)。
    """
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

        if name != "unknown":
            attendance_manager.update_attendance(name, camera_id)

        # 人領域の描画を行いNameとSimilarityを描く関数（オプション）
        draw_results(frame, ltrb, track_id, name, similarity)

def draw_results(frame, ltrb, track_id, name, similarity):
    """
    検出結果をフレームに描画する。

    :param frame: numpy.ndarray
        描画対象の画像フレーム。
    :param ltrb: list of int
        バウンディングボックスの左上 (left, top) と右下 (right, bottom) 座標。
    :param track_id: int
        トラッキングID。
    :param name: str
        認識された名前。
    :param similarity: float
        類似度スコア。
    """
    cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {track_id}, Name: {name}",
                (int(ltrb[0]), int(ltrb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 類似度を右下に描画
    h, w = frame.shape[:2]
    similarity_text = f"Similarity: {similarity:.2f}"
    text_x = w - 200
    text_y = h - 10
    cv2.putText(frame, similarity_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    """
    リアルタイムの顔認識と入退室管理を開始するメイン関数。
    """
    person_detector, face_recognizer, tracker, attendance_manager = initialize_modules()
    tracked_names = {}

    camera_id = 0  # 入室カメラIDとして仮定
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame, person_detector, face_recognizer, tracker, attendance_manager, tracked_names, camera_id)

            cv2.imshow("Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # 全ユーザーの状態を表示
        print("\n[INFO] カメラ停止後の全ユーザー状態を表示:")
        attendance_manager.show_all_status()

if __name__ == "__main__":
    main()
