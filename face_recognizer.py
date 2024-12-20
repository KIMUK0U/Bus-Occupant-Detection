from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class FaceRecognizer:
    def __init__(self, known_face_encodings, known_face_names):
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names

    def recognize(self, face_encoding):
        similarities = cosine_similarity([face_encoding], self.known_face_encodings)
        best_match_index = np.argmax(similarities)
        if similarities[0, best_match_index] > 0.5:  # 類似度のしきい値
            return self.known_face_names[best_match_index]
        return "Unknown"
