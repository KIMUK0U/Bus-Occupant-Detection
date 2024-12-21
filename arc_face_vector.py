from insightface.app import FaceAnalysis
import cv2
# モデルの初期化
app = FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU (-1) または GPU (0, 1, ...)

# 画像から顔を解析
img = cv2.imread("face/output_jpg/myface02.jpg")
faces = app.get(img)

# 特徴ベクトルを取得
for face in faces:
    embedding = face.normed_embedding
    print(embedding)
