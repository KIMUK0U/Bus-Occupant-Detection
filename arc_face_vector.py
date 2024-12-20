import torch
import torchvision.transforms as transforms
from PIL import Image
from insightface import model_zoo

def load_model():
    """
    ArcFaceの事前学習済みモデルをロードします。
    """
    model = model_zoo.get_model('arcface_r100_v1')
    model.prepare(ctx_id=-1)  # -1はCPU, 0以上はGPU
    return model

def preprocess_image(image_path):
    """
    入力画像を前処理します。
    """
    transform = transforms.Compose([
        transforms.Resize((112, 112)),  # ArcFaceは112x112の画像が必要
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # -1～1に正規化
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # バッチ次元を追加

def get_face_vector(image_path, model):
    """
    入力画像から特徴ベクトルを取得します。
    """
    # 画像の前処理
    preprocessed_image = preprocess_image(image_path)
    
    # モデル推論
    with torch.no_grad():
        embedding = model.get_embedding(preprocessed_image)
    
    # 512次元ベクトルを返す
    return embedding.numpy()

if __name__ == "__main__":
    # デモ用コード
    import sys
    if len(sys.argv) < 2:
        print("Usage: python arc_face_vector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model = load_model()
    vector = get_face_vector(image_path, model)
    print("Feature Vector:")
    print(vector)
