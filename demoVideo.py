# demoVideo.py
import cv2
from person_detector import PersonDetector
from face_recognizer import FaceRecognition
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
from real_time_face_recognition import initialize_modules, detect_and_recognize_faces, process_frame, draw_results
from config import VIDEO_PATH


def main():
    video_path = VIDEO_PATH  # デモ動画のパス
    person_detector, face_recognizer, tracker, attendance_manager = initialize_modules()
    tracked_names = {}
    camera_id = 0 # ドア正面と定義、入室を検知する

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    # ---【追加】 出力動画ファイルの準備 -------------------------------
    # ※ 出力パスやFPS、画質などは適宜変更してください
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4形式
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_demo9.mp4", fourcc, fps, (width, height))
    # -----------------------------------------------------------------

    count=0
    while cap.isOpened():
        count +=1
        if count % 1 == 0:

            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame, person_detector, face_recognizer, tracker, attendance_manager, tracked_names, camera_id)
            
            # ---【追加】 保存用に書き出し ----------------------------
            out.write(frame)
            # ------------------------------------------------------

            cv2.imshow("Video Demo - Real-Time Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 連続で読み進めるだけ
            _ = cap.read()

    cap.release()
    # ---【追加】 VideoWriterを必ず解放する ----
    out.release()
    # -----------------------------------------
    cv2.destroyAllWindows()
    # 全ユーザーの状態を表示
    print("\n[INFO] カメラ停止後の全ユーザー状態を表示:")
    attendance_manager.show_all_status()


if __name__ == "__main__":
    main()
