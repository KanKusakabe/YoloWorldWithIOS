# YOLO‑World iOS 組み込みガイド

> **注記** : ここで紹介する手順では **Core ML へエクスポートする際にクラスラベルを固定** します。そのため Core ML 化されたモデルでは、YOLO‑World 本来の **オープン語彙（自然言語プロンプト）機能は利用できません**。

---

## 目次

1. [概要](#概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [前提条件](#前提条件)
4. [データセット準備](#データセット準備)
5. [モデル学習 / ファインチューニング](#モデル学習--ファインチューニング)
6. [Core ML へのエクスポート](#core-ml-へのエクスポート)
7. [Xcode への組み込み](#xcode-への組み込み)
8. [Vision を用いた推論](#vision-を用いた推論)
9. [パフォーマンス向上 Tips](#パフォーマンス向上tips)
10. [トラブルシューティング](#トラブルシューティング)
11. [ライセンス](#ライセンス)

---

## 概要

YOLO‑World は YOLO 系の高速物体検出と **視覚 + 言語エンコーダ** を組み合わせたモデルです。本ガイドでは **Ultralytics 版 YOLO‑World** を学習させ、Core ML 形式に変換して **iOS デバイス上でオフライン推論** できるようにする手順をまとめます。

> Core ML は動的な文字列入力をサポートしていないため、エクスポート時にクラスを `model.set_classes([...])` で固定します。

## アーキテクチャ

```
カメラフレーム → Vision (VNImageRequestHandler) → Core ML (YOLO‑World.mlmodel)
                                                             ↓
                                              バウンディングボックス + クラスラベル
```

- 学習: **PyTorch + Ultralytics YOLO‑World**
- 変換: **coremltools ≥ 7** で `.mlpackage` 出力
- 推論: Vision フレームワーク経由で SwiftUI/UKit に描画

## 前提条件

| ステージ        | バージョン / ツール                       |
| ----------- | --------------------------------- |
| Python      | 3.9 – 3.12                        |
| PyTorch     | ≥ 2.2                             |
| Ultralytics | ≥ 8.1 (`pip install ultralytics`) |
| coremltools | ≥ 7.0                             |
| Xcode       | 15+                               |
| iOS         | 17+ (Neural Engine 推奨)            |

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics coremltools
```

## データセット準備

1. 目的のクラスを含む画像を収集
2. **YOLO 形式** でアノテーション (`images/`, `labels/`)
3. `data.yaml` を作成

```yaml
path: .
train: images/train
val:   images/val
names:
  0: cup
  1: keyboard
  2: mouse
```

## モデル学習 / ファインチューニング

```python
from ultralytics import YOLO
model = YOLO("yolov8s-world.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, device=0)
model.set_classes(["cup", "keyboard", "mouse"])
model.save("custom_yoloworld.pt")
```

## Core ML へのエクスポート

```python
from ultralytics import YOLO
model = YOLO("custom_yoloworld.pt")
model.export(format="coreml", nms=True, int8=False)
```

- `nms=True` で NMS をモデル内に組み込み
- `int8=True` で軽量化も可能

`YOLO‑World.mlpackage` を Xcode にドラッグ & ドロップすれば準備完了です。

## Xcode への組み込み

1. **モデル追加** : `.mlpackage` をプロジェクトに入れる
2. **カメラ権限** : `Info.plist` に `NSCameraUsageDescription` を追加
3. **Swift コード例**

```swift
import Vision
import CoreML

let mlModel = try YOLO_World(configuration: .init()).model
let vnModel = try VNCoreMLModel(for: mlModel)
```

## Vision を用いた推論

```swift
class Detector {
    let request: VNCoreMLRequest
    init() {
        let mlModel = try! YOLO_World(configuration: .init()).model
        let vnModel = try! VNCoreMLModel(for: mlModel)
        request = VNCoreMLRequest(model: vnModel) { req, _ in
            if let results = req.results as? [VNRecognizedObjectObservation] {
                // バウンディングボックス処理
            }
        }
    }
    func process(pixelBuffer: CVPixelBuffer) {
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer).perform([request])
    }
}
```

## パフォーマンス向上 Tips

| Tip                 | 効果                                  |
| ------------------- | ----------------------------------- |
| `yolov8s-world` を使用 | A17/M2 端末で 20–40 FPS                |
| `int8=True` で書き出し   | 容量 40 % 減、FPS 若干向上                  |
| `computeUnits` 切替   | `.all`, `.cpuAndNeuralEngine` などテスト |

## トラブルシューティング

- **検出結果が出ない**: `request.imageCropAndScaleOption = .centerCrop` を試す
- **エクスポート失敗**: `coremltools` のバージョン確認。カスタムレイヤーが残っていないかチェック
- **アプリサイズ肥大**: INT8 化 & 不要言語リソース削除

## ライセンス

YOLO‑World は Apache 2.0、 本ガイドは MIT License です。

---

# YOLO-World iOS Integration Guide (English)

> **Note** : This workflow **fixes class labels at export time**. The original "open-vocabulary" capability of YOLO-World is **not preserved** once the model is converted to Core ML.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Training / Fine-Tuning](#model-training--fine-tuning)
6. [Export to Core ML](#export-to-core-ml)
7. [Integrating the Model into Xcode](#integrating-the-model-into-xcode)
8. [Running Inference with Vision](#running-inference-with-vision)
9. [Performance Tips](#performance-tips)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Overview

YOLO-World combines the speed of the YOLO family with a dual **vision + language** encoder to enable natural-language object detection. For iOS deployment we convert a custom-trained YOLO-World model to **Core ML** so it can run on-device (Neural Engine / GPU / CPU).

Because Core ML does not support dynamic language prompts, we **freeze a finite set of classes** (`model.set_classes([...])`) before exporting.

## Architecture

```
Camera Frame → Vision (VNImageRequestHandler) → Core ML (YOLO-World.mlmodel)
                                                       ↓
                                    Bounding Boxes + Class Labels
```

- **Training side** : PyTorch + Ultralytics YOLO-World
- **Export side** : coremltools ≥ 7 generates an `.mlpackage`
- **Runtime side** : Vision framework wraps the model; results rendered with SwiftUI / UIKit

## Prerequisites

| Host Stage  | Version / Tool                           |
| ----------- | ---------------------------------------- |
| Python      | 3.9 – 3.12                               |
| PyTorch     | ≥ 2.2                                    |
| Ultralytics | ≥ 8.1 (`pip install ultralytics`)        |
| coremltools | ≥ 7.0                                    |
| Xcode       | 15 or later                              |
| iOS         | 17+ (Neural Engine strongly recommended) |

```bash
# minimal environment setup
python -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics coremltools
```

## Dataset Preparation

1. Collect images for the target classes you care about.
2. Label them in **YOLO format** (`images/train`, `labels/train` etc.). Tools: [Roboflow](https://roboflow.com/), [CVAT](https://github.com/opencv/cvat), labelImg.
3. Create a `data.yaml`:
   ```yaml
   path: .  # workspace root
   train: images/train
   val:   images/val
   names:
     0: cup
     1: keyboard
     2: mouse
   ```

## Model Training / Fine-Tuning

```python
from ultralytics import YOLO

# 1. Load a pretrained YOLO-World backbone
model = YOLO("yolov8s-world.pt")  # choose s/m/l depending on device budget

# 2. Fine-tune (transfer learning)
model.train(data="data.yaml", epochs=50, imgsz=640, batch=16, device=0)

# 3. Freeze class list (open-vocabulary ➔ closed-set)
model.set_classes(["cup", "keyboard", "mouse"])
model.save("yoloworld_custom.pt")
```

## Export to Core ML

```python
from ultralytics import YOLO
model = YOLO("yoloworld_custom.pt")
model.export(format="coreml", nms=True, int8=False)  # outputs YOLO-World.mlpackage
```

- `nms=True` embeds Non-Max Suppression into the model.
- Set `int8=True` for smaller size at some accuracy cost.

The resulting `` is ready for Xcode (drag-and-drop into the project navigator).

## Integrating the Model into Xcode

1. **Add the model**: Drag `YOLO-World.mlpackage` into Xcode → *Copy resources if needed*.
2. **Enable fast compute**:
   - Project → *Signing & Capabilities* → *App Sandbox* → *No extra entitlement required* – Core ML/Vision is sandbox-safe.
3. **Update Info.plist** – add `NSCameraUsageDescription`.
4. **Import Vision in Swift code**:
   ```swift
   import Vision
   import CoreML
   ```

## Running Inference with Vision

```swift
class Detector {
    private let model: VNCoreMLModel
    private let request: VNCoreMLRequest

    init() {
        let mlModel = try! YOLO_World(configuration: MLModelConfiguration()).model
        model = try! VNCoreMLModel(for: mlModel)
        request = VNCoreMLRequest(model: model) { req, _ in
            guard let results = req.results as? [VNRecognizedObjectObservation] else { return }
            DispatchQueue.main.async {
                // handle bounding boxes & labels here
            }
        }
        request.imageCropAndScaleOption = .scaleFill
    }

    func process(pixelBuffer: CVPixelBuffer) {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
}
```

Use `VNRecognizedObjectObservation.boundingBox` to draw overlays in SwiftUI or UIKit.

## Performance Tips

| Tip                | Effect                                               |
| ------------------ | ---------------------------------------------------- |
| Use `` (small)     | 20-40 FPS on A17 / M2 iPhones                        |
| `int8=True` export | Reduces size \~40 % / boosts FPS \~10 %              |
| `computeUnits`     | Test `.all`, `.cpuAndGPU`, `.cpuAndNeuralEngine`     |
| Batch inference    | Not supported by Vision; process frames individually |

## Troubleshooting

- **Empty detections on M-series simulator** : Run on real device or set `request.imageCropAndScaleOption = .centerCrop`.
- **Core ML export fails** : Update `coremltools`; verify no custom ops remain.
- **App size too large** : Remove unused language resources from `.mlpackage`, or convert to INT8.

## License

YOLO-World is released under Apache 2.0. This guide is released under the MIT License.

