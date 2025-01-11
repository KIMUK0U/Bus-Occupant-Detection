# 入退室管理システム

このプロジェクトは、顔認識を用いた入退室管理システムです。  
YOLOv8 を利用して人物を検出し、InsightFace の FaceAnalysis モジュールを使用して顔認証を行います。  
認識されたユーザーは、入室・退室ログとして記録され、後で確認や通知に使用されます。

このプロジェクトは InsightFace ライブラリの FaceAnalysis モジュールを使用して、顔認識を実現しています。

## セットアップ

### 必要な環境
- Python 3.8 以上
- NVIDIA Jetson Orin Nano (または GPU 搭載マシン)
- 必要なライブラリ

### 依存ライブラリ
以下のコマンドで必要なライブラリをインストールしてください：

```bash
pip install -r requirements.txt
```
### 3. **使用方法 (Usage)**
プログラムの実行方法を具体的に説明します。サンプルコマンドや入力・出力ファイルの説明を含めると良いです。

#### 記載例
```markdown
## 使用方法

1. **ユーザー登録**
   - 入退室管理システムでは、顔認証を行うために事前にユーザー登録を行います。
   - ユーザー名と対応する顔画像を `data/faces/<ユーザー名>/` フォルダに保存してください。

2. **プログラムの実行**
   - 動画ファイルを使用した入退室ログ生成：
     ```bash
     python demoVideo.py
     ```
   - リアルタイムカメラ入力：
     ```bash
     python live_demo.py
     ```

3. **出力**
   - 入退室ログは `path file name` に保存されます。
   - ログの形式:
     ```
     名前, 日時, 状態 (入室/退室)
     ```

4. **設定**
   - 動画パスやモデルパスなどは `config.py` で変更可能です。
   - そのほかに、顔認証事前登録用顔画像は`real_time_face_recognition.py`に含まれるinitialize_modules関数で変更可能です。


## 使用ライブラリ
- InsightFace (Apache License 2.0)
  - GitHub: [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

- YOLOv8 by Ultralytics (GPL-3.0 License)
  - GitHub: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## ライセンスについて
本プロジェクトは以下のライセンスに基づいています：
- InsightFace: Apache License 2.0
- YOLOv8: GPL-3.0 License

本プロジェクトの一部には、Apache License 2.0 のもとで公開されたソフトウェア (InsightFace) を利用しています。
また、YOLOv8 を利用するため、当プロジェクトは GPL-3.0 に準拠しています。
詳細については、各ライブラリの公式ページをご参照ください。


## 学習データセットについて
InsightFace のモデルは、WIDER FACE など多様な顔画像データセットを使用して学習されています。
これにより、さまざまな環境や人種に対して高い汎用性を持っています。

## 商用利用に関する注意
このプロジェクトに含まれるソフトウェアおよびライブラリは、Apache License 2.0 の条件を満たす限り、商用利用が可能です。
ただし、商用利用の場合は、InsightFace および関連データセットのライセンス条件を再確認することを推奨します。
