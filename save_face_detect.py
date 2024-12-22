import cv2
from face_detector import FaceDetector
from person_detector import PersonDetector

def save_faces_from_image(input_image_path, output_directory, face_confidence_threshold=0.5):
    # モジュールの初期化
    face_detector = FaceDetector()
    person_detector = PersonDetector()

    # 入力画像を読み込む
    image = cv2.imread(input_image_path)
    if image is None:
        print("画像を読み込めませんでした。")
        return

    # 人の検出
    persons = person_detector.detect(image)
    print(f"検出された人物: {len(persons)}")

    # 各人について顔を検出
    face_count = 0
    for idx, person in enumerate(persons):
        bbox, _, _ = person
        x1, y1, x2, y2 = bbox

        # 検出された人の領域を切り抜き
        cropped_person = image[y1:y2, x1:x2]

        # 顔を検出
        faces = face_detector.detect(cropped_person)

        for face in faces:
            fx1, fy1, fx2, fy2, confidence = face
            if confidence < face_confidence_threshold:
                continue
            fx1 -= 220
            fy1 -= 320
            fx2 += 120
            fy2 += 80
            # 顔領域を切り抜き
            cropped_face = cropped_person[fy1:fy2, fx1:fx2]

            # 保存
            output_path = f"{output_directory}/face_{idx}_{face_count}.jpg"
            cv2.imwrite(output_path, cropped_face)
            print(f"顔画像を保存: {output_path}")
            face_count += 1

    print(f"合計{face_count}枚の顔画像を保存しました。")

if __name__ == "__main__":
    input_image_path = "data/SampleJpg/user1face1.jpg"  # 入力画像のパス
    output_directory = "data/faces/user1"      # 顔画像の保存先ディレクトリ

    # ディレクトリが存在しない場合は作成
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    save_faces_from_image(input_image_path, output_directory)
