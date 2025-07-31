"""
Yolo Worldで画像を予測するコード
"""

from ultralytics import YOLO

# 1) YOLO-World のモデルを読み込む
model = YOLO("yolov8m-world.pt")
model.set_classes(["cup", "keyboard", "mouse", "helmet"])

image_path = "test.png"  # 予測したい画像のパスを指定

# 2) 画像を予測し、バウンディングボックス付き画像を自動保存
results = model.predict(source=image_path, save=True, save_txt=True)

# 保存先は runs/detect/predict フォルダになります
print("保存先:", results[0].save_dir)

