# config.py

# 動画の設定
VIDEO_PATH = "data\\VideoAfterEdit\\Beside\\leaveBesideDoor6.MOV"  # 入力動画ファイルのパス

# モデルの設定
PERSON_DETECTOR_MODEL_PATH = "yolov8n.pt"  # 人物検出モデル

# ログの設定
LOG_DIRECTORY = "logs/"  # ログファイル保存先
LOG_FILE = "entry_logs.csv"  # ログファイル名

# その他の設定
DETECTION_INTERVAL = 5  # フレーム間隔（5フレームごとに処理を実行）
SIMILARITY_THRESHOLD = 0.5  # 顔認証の類似度閾値
