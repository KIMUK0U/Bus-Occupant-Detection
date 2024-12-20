import cv2
from face_detector import FaceDetector

def main():
    # FaceDetectorを初期化
    face_detector = FaceDetector("face/FaceDetector/face_detection_yunet_2023mar.onnx")
    
    # テスト用画像のパス
    image_path = "face/output_jpg/myface02.jpg"  # 検証したい画像ファイルを指定
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image '{image_path}'")
        return

    # 顔検出を実行
    faces = face_detector.detect(image)

    # 検出結果を描画
    for x1, y1, x2, y2, confidence in faces:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            image,
            f"Face: {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    # 表示画像をリサイズ（幅800ピクセル、高さは比率に応じて調整）
    height, width = image.shape[:2]
    scale = 800 / width
    resized_image = cv2.resize(image, (800, int(height * scale)))

    # 結果を表示
    cv2.imshow("Face Detection Result", resized_image)

    # キーを押して終了
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
