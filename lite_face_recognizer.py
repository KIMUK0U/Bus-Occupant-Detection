import cv2
from paddle.inference import Config, create_predictor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mtcnn import MTCNN

class LightweightFaceRecognition:
    def __init__(self, model_dir):
        config = Config(f"{model_dir}/inference.pdmodel", f"{model_dir}/inference.pdiparams")
        config.disable_gpu()
        config.enable_memory_optim()
        self.predictor = create_predictor(config)
        self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        self.output_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.database = {}
        self.face_detector = cv2.FaceDetectorYN.create('face_detection_yunet_2023mar.onnx', "", (320, 320), 0.6, 0.3, 5000)


    def preprocess_face(self, image, box):
        x, y, width, height = box
        x, y, width, height = int(x), int(y), int(width), int(height)
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = np.asarray(face, dtype="float32") / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def predict_embedding(self, face):
        self.input_handle.copy_from_cpu(face)
        self.predictor.run()
        embedding = self.output_handle.copy_to_cpu()
        return embedding

    def detect_faces(self, image):
        detections = self.face_detector.detect_faces(image)
        if len(detections) == 0:
            return None
        return detections

    def enroll_person(self, person_id, image_paths):
        embeddings = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}")
                continue

            detections = self.detect_faces(image)
            if not detections:
                print(f"No face detected in {image_path}")
                continue

            for detection in detections:
                box = detection['box']
                preprocessed_face = self.preprocess_face(image, box)
                embedding = self.predict_embedding(preprocessed_face)
                embeddings.append(embedding)

        self.database[person_id] = embeddings
        print(f"Registered {len(embeddings)} embeddings for person ID '{person_id}'")

    def enroll_multiple_persons(self, persons_data):
        """
        複数のユーザーを一括登録
        """
        for person_id, image_paths in persons_data.items():
            print(f"Registering person ID '{person_id}' with {len(image_paths)} images...")
            self.enroll_person(person_id, image_paths)

    def verify_person(self, input_image, person_id, threshold=0.6):
        detections = self.detect_faces(input_image)
        if not detections:
            return "No face detected in the input image.", 0.0

        for detection in detections:
            box = detection['box']
            preprocessed_face = self.preprocess_face(input_image, box)
            input_embedding = self.predict_embedding(preprocessed_face)

            if person_id not in self.database:
                return f"Person ID '{person_id}' not found in database.", 0.0

            registered_embeddings = self.database[person_id]
            if not registered_embeddings:
                return f"No embeddings found for Person ID '{person_id}'.", 0.0

            similarities = [
                cosine_similarity([input_embedding.flatten()], [embedding.flatten()])[0][0]
                for embedding in registered_embeddings
            ]
            max_similarity = max(similarities) if similarities else 0

            if max_similarity > threshold:
                return f"Match with similarity {max_similarity:.2f}", max_similarity

        return "No match found.", 0.0


# 使用例
if __name__ == "__main__":
    model_dir = "Bus_Occupant_Detect/mobileface_v1.0_infer"
    face_recognizer = LightweightFaceRecognition(model_dir)

    # 登録データ
    persons_data = {
        "KIMURA": ["data/faces/user1/face_0.jpg", "data/faces/user1/face_1.jpg"],
        "KOIKE": ["data/faces/user2/face_0.jpg", "data/faces/user2/face_1.jpg"]
    }

    # ユーザー登録
    for person_id, image_paths in persons_data.items():
        face_recognizer.enroll_person(person_id, image_paths)

    # テスト用の顔認証
    test_image = cv2.imread("data/faces/user1/face_2.jpg")
    result, similarity = face_recognizer.verify_person(test_image, "KIMURA")
    print(result, similarity)
