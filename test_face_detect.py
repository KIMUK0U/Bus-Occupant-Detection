import cv2 as cv

# モデルの読み込み
face_detector = cv.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', "", (320, 320), 0.6, 0.3, 5000)

# 入力画像の読み込み
image = cv.imread('data\\faces\\user1\\face_1.jpg')

# 入力サイズの設定
height, width, _ = image.shape
face_detector.setInputSize((width, height))

# 顔検出の実行
_, faces = face_detector.detect(image)

# 検出結果の表示
if faces is not None:
    for face in faces:
        # バウンディングボックスの描画
        bbox = list(map(int, face[:4]))
        cv.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    cv.imshow('Detected Faces', image)
    cv.waitKey(0)
else:
    print("顔が検出されませんでした。")
