import cv2
from person_detector import PersonDetector
from lite_face_detecter import FaceDetector
from lite_face_recognizer import LightweightFaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from config import PERSON_DETECTOR_MODEL_PATH


class RealTimeFaceRecognition:
    def __init__(self, person_model_path, face_model_dir):
        """
        初期化：各モジュール（人物検出、顔検出、顔認証、トラッキング）を初期化し、
        顔認証用のデータベースを登録。
        """
        self.person_detector = PersonDetector(model_path=person_model_path)
        self.face_detector = FaceDetector()
        self.face_recognizer = LightweightFaceRecognition(face_model_dir)
        self.tracker = DeepSort(max_age=30, nms_max_overlap=1.0, max_cosine_distance=0.7)
        self.tracked_names = {}  # トラッキングIDに対応する名前を保存

        # 顔画像データの登録
        self.enroll_persons()

    def enroll_persons(self):
        """
        登録データを用いて顔認証用データベースを構築。
        """
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
                "data/faces/user2/face_1.jpg"
            ],
            "YASUI": [
                "data/faces/user3/face_0.jpg"
            ],
            "KUROZAWA": [
                "data/faces/user4/face_0.jpg",
                "data/faces/user4/face_1.jpg"
            ],
            "HOJO": [
                "data/faces/user5/face_0.jpg"
            ]
        }
        self.face_recognizer.enroll_multiple_persons(persons_data)

    def detect_and_track(self, frame):
        """
        人物を検出し、トラッキングを行う。
        """
        person_detections = self.person_detector.detect(frame)
        formatted_detections = [
            (bbox, confidence, class_id) for bbox, confidence, class_id in person_detections
        ]
        tracks = self.tracker.update_tracks(formatted_detections, frame=frame)
        return tracks

    def process_face_recognition(self, person_roi, track_id):
        """
        顔検出と顔認証を行い、結果を返す。
        """
        if track_id in self.tracked_names and self.tracked_names[track_id] != "Unknown":
            # すでに名前が判明している場合、認証をスキップ
            return self.tracked_names[track_id], 1.0

        face_boxes = self.face_detector.detect_faces(person_roi)
        if not face_boxes:
            print(f"No faces detected in track ID: {track_id}")
            return "Unknown", 0.0

        best_match = ("Unknown", 0.0)
        for face_box in face_boxes:
            preprocessed_face = self.face_recognizer.preprocess_face(person_roi, face_box)
            embedding = self.face_recognizer.predict_embedding(preprocessed_face)

            for person_id, embeddings in self.face_recognizer.database.items():
                similarities = [
                    cosine_similarity([embedding.flatten()], [db_embedding.flatten()])[0][0]
                    for db_embedding in embeddings
                ]
                max_similarity = max(similarities) if similarities else 0
                if max_similarity > best_match[1]:
                    best_match = (person_id, max_similarity)

        if best_match[1] > 0.6:  # 閾値を超えた場合に名前を記録
            self.tracked_names[track_id] = best_match[0]
        else:
            print(f"Not matched. {best_match[0]}, similarity:{best_match[1]}")

        return best_match

    def process_frame(self, frame):
        """
        フレームごとに処理を行い、検出結果を描画。
        """
        tracks = self.detect_and_track(frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            person_roi = frame[int(ltrb[1]):int(ltrb[3]), int(ltrb[0]):int(ltrb[2])]

            if person_roi.size == 0:
                print("Error: person_roi is empty or invalid!")
                continue

            name, similarity = self.process_face_recognition(person_roi, track_id)
            print(f"Match {name}. similarity: {similarity}")
            self.draw_results(frame, ltrb, track_id, name, similarity)

    @staticmethod
    def draw_results(frame, ltrb, track_id, name, similarity):
        """
        フレームに結果を描画する。
        """
        cv2.rectangle(frame, (int(ltrb[0]), int(ltrb[1])), (int(ltrb[2]), int(ltrb[3])), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}, Name: {name}", (int(ltrb[0]), int(ltrb[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Similarity: {similarity:.2f}", (int(ltrb[0]), int(ltrb[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def run_demo_video(self, video_path, output_path="output.mp4"):
        """
        動画を使用してデモ処理を行い、結果を保存。
        """
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if out is None:
                height, width = frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            self.process_frame(frame)
            out.write(frame)

            cv2.imshow("Demo Video - Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    person_model_path = PERSON_DETECTOR_MODEL_PATH
    face_model_dir = "Bus_Occupant_Detect/mobileface_v1.0_infer"
    video_path = "data\\VideoAfterEdit\\Beside\\IMG_1956.MOV"

    face_recognition_system = RealTimeFaceRecognition(person_model_path, face_model_dir)

    face_recognition_system.run_demo_video(video_path, output_path="output.mp4")
