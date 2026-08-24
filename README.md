# バス乗降検知・置き去り防止システム

送迎バス車内での児童置き去り事故を防ぐことを目的とした、顔認証による乗降管理システムです。
既存の人感センサーは「車内に誰かがいる」ことしか示せず、警報が誤作動と判断されて確認行動につながらないことがあります。
そこで「誰が残っているか」を個人単位で特定できるよう、顔認証を採用しました。

小型コンピュータ（NVIDIA Jetson Orin Nano）で動かす必要があるため、まず YOLOv8 で人物を検出し、
人物が写っているフレームに対してのみ InsightFace による顔認証を実行する構成にすることで、
限られた計算資源でもリアルタイムに近い処理を実現しています。認識結果は乗降履歴として記録されます。

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

## 使用方法

1. **ユーザー登録**
   - 顔認証を行うために、事前にユーザー登録を行います。
   - `real_time_face_recognition.py` の `initialize_modules()` 内にある `persons_data` に、ユーザー名と対応する顔画像のパスを追加してください。

2. **プログラムの実行**
   - 動画ファイルを使った検証（トラッキング結果を描画した動画を書き出す、フル機能版）：
     ```bash
     python demoVideo.py
     ```
   - Jetson等の小型コンピュータ向け軽量版（`realtime`／`demo` を実行時に選択）：
     ```bash
     python lite_main.py
     ```

3. **出力**
   - 乗降ログは `config.py` の `LOG_DIRECTORY` / `LOG_FILE`（既定では `logs/entry_logs.csv`）に保存されます。
   - ログの形式:
     ```
     名前, 日時, 状態 (入室/退室)
     ```

4. **設定**
   - 動画パスやモデルパスなどは `config.py` で変更可能です。
   - 顔認証の事前登録用データは、`real_time_face_recognition.py` に含まれる `initialize_modules()` 関数で変更可能です。


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
