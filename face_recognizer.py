import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognition:
    def __init__(self, ctx_id=-1):
        """
        FaceRecognition の初期化。
        :param ctx_id: GPU (-1 for CPU) を指定。
        """
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id)  # モデルの準備
        self.database = {}  # 登録データベース

    def enroll_person(self, person_id, image_paths):
        """
        特定の個人の画像を登録し、特徴ベクトルを保存する。
        :param person_id: 個人を識別するID。
        :param image_paths: 登録する画像のパスのリスト。
        """
        embeddings = []
        for image_path in image_paths:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not read image {image_path}")
                continue

            faces = self.app.get(img)
            if len(faces) == 0:
                print(f"No face detected in {image_path}")
                continue

            for face in faces:
                embeddings.append(face.normed_embedding)

        self.database[person_id] = embeddings
        print(f"Registered {len(embeddings)} embeddings for person ID '{person_id}'")
    
    def enroll_multiple_persons(self, persons_data):
        """
        複数の個人を登録する。
        :param persons_data: 辞書形式で登録するデータ。{person_id: [image_paths], ...}
        """
        for person_id, image_paths in persons_data.items():
            print(f"Registering person ID '{person_id}' with {len(image_paths)} images...")
            self.enroll_person(person_id, image_paths)

    def verify_person(self, input_image, person_id, threshold=0.5):
        """
        入力画像が指定された個人の登録画像と一致するか確認する。
        :param input_image: 認証する画像（numpy.ndarray）。
        :param person_id: 照合する登録済みの個人ID。
        :param threshold: 類似度の閾値。
        :return: 照合結果のメッセージ。
        """
        if input_image is None:
            raise ValueError("Error: Input image is None.")

        faces = self.app.get(input_image)  # jetson orin nanoで数十ミリ秒程度の推論時間
        if len(faces) == 0:
            return "No face detected in the input image."

        input_embedding = faces[0].normed_embedding
        if person_id not in self.database:
            return f"Person ID '{person_id}' not found in database."

        registered_embeddings = self.database[person_id]
        if not registered_embeddings:
            return f"No embeddings found for Person ID '{person_id}'."

        similarities = [cosine_similarity([input_embedding], [embedding])[0][0] for embedding in registered_embeddings]
        max_similarity = max(similarities) if similarities else 0

        if max_similarity > threshold:
            return f"Match with similarity {max_similarity:.2f}"
        else:
            return f"No match (highest similarity: {max_similarity:.2f})"
