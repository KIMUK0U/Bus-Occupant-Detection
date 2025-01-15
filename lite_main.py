import cv2
from lite_face_recognizer import LightweightFaceRecognition
from lite_face_detecter import FaceDetector
from person_detector import PersonDetector
from lite_real_time_face_recognition import RealTimeFaceRecognition

if __name__ == "__main__":
    person_model_path = "path/to/yolov8_model.pt"
    face_model_dir = "Bus_Occupant_Detect/mobileface_v1.0_infer"
    video_path = "data/VideoAfterEdit/Front/IMG_1944.MOV"  # デモ動画のパス

    face_recognition_system = RealTimeFaceRecognition(person_model_path, face_model_dir)

    # 実行モードを選択
    mode = input("Select mode (realtime/demo): ").strip().lower()
    if mode == "realtime":
        face_recognition_system.run_realtime()
    elif mode == "demo":
        face_recognition_system.run_demo_video(video_path)
    else:
        print("Invalid mode selected.")
