# requirements:
#   pip install ultralytics coremltools torch torchvision

from ultralytics import YOLO

# 1) YOLO-World のモデルを読み込む
# モデル一覧：https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes
model = YOLO("yolov8m-world.pt")

# 2) 使いたいクラスを固定（※オープン語彙は使わない）
classes = ["cup", "keyboard", "mouse", "helmet"]
model.set_classes(classes)

# 3) 出力ファイル名
# <base model name>_<classes>.mlpackage
file_name = f"yolov8m-world_{'_'.join(classes)}.mlpackage"

# 4) Core ML 形式で書き出し
#    - nms=True   : モデル内部で Non-Max Suppression を実行
#    - int8=False : 精度重視（軽量化したい場合は True）
mlpackage_path = model.export(format="coreml", nms=True, int8=False)

print(f"✅ Core ML package saved to: {mlpackage_path}")
