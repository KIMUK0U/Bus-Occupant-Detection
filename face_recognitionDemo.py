from face_recognition_module import FaceRecognition

"""
顔認証モジュールを試すためのデモプログラム
"""

def main():
    # FaceRecognition インスタンスの作成
    face_recognition = FaceRecognition(ctx_id=-1)  # -1 for CPU, 0 for GPU

    # 登録フェーズ
    person_id = "user1234"
    image_paths = [
        "face/output_jpg/myface01.jpg",
        "face/output_jpg/myface02.jpg",
        "face/output_jpg/myface03.jpg"
    ]
    face_recognition.enroll_person(person_id, image_paths)

    # 認証フェーズ
    test_image_path = "face/output_jpg/test_face.jpg"
    result = face_recognition.verify_person(test_image_path, person_id)
    print(result)

if __name__ == "__main__":
    main()