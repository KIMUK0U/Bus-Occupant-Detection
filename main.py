import cv2
from person_detector import PersonDetector
from face_detector import FaceDetector
from face_recognizer import FaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_modules():
    """モジュールの初期化を行う"""
    person_detector = PersonDetector()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognition()
    tracker = DeepSort()

    # ユーザー登録
    # 登録データの辞書
    persons_data = {
        "KIMURA KOTA": [
            "data/faces/user1/face_0.jpg",
            "data/faces/user1/face_1.jpg"
        ],
        "KOIKE YUTA": [
            "data/faces/user2/face_0.jpg"
        ],
        "YASUI": [
            "data/faces/user3/face_0.jpg"
        ],
        "KUROZAWA": [
            "data/faces/user4/face_0.jpg",
            "data/faces/user4/face_1.jpg"
        ]
    }
    # 一括登録
    face_recognizer.enroll_multiple_persons(persons_data)

    return person_detector, face_detector, face_recognizer, tracker

def detect_and_recognize_faces(person_roi, face_detector, face_recognizer, track_id, tracked_names):
    """`person_roi` から顔検出と認識を行う"""
    if track_id in tracked_names and tracked_names[track_id] != "unknown":
        return tracked_names[track_id]

    faces = face_detector.detect(person_roi)
    name = "unknown"

    if faces:
        print(f"Detected {len(faces)} face(s) in person ROI")
        h, w, _ = person_roi.shape
        for x1, y1, x2, y2, confidence in faces:
            # 座標の範囲を調整
            margin = 100
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            face_roi = person_roi[y1:y2, x1:x2]

            if face_roi is None or face_roi.size == 0:
                print("Error: face_roi is empty!")
                continue

            # 顔認証
            for person_id in face_recognizer.database.keys():
                print("Mark3")
                result = face_recognizer.verify_person(face_roi, person_id)
                print(f"Verification result for {person_id}: {result}")
                if "Match" in result:
                    name = person_id
                    break

            if name != "unknown":
                tracked_names[track_id] = name
                break

    return name

def process_frame(frame, person_detector, face_detector, face_recognizer, tracker, tracked_names):
    """フレームごとに処理を行う"""
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
        if person_roi is None or person_roi.size == 0:
            print("Error: person_roi is empty or invalid!")
            name = "Unknown"
        else:
            name = detect_and_recognize_faces(person_roi, face_detector, face_recognizer, track_id, tracked_names)

        draw_results(frame, ltrb, track_id, name)

def draw_results(frame, ltrb, track_id, name):
    """描画を行う"""
    cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
    cv2.putText(frame, f"ID: {track_id}, Name: {name}",
                (int(ltrb[0]), int(ltrb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def main():
    person_detector, face_detector, face_recognizer, tracker = initialize_modules()
    tracked_names = {}

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame, person_detector, face_detector, face_recognizer, tracker, tracked_names)

        cv2.imshow("Real-Time Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
