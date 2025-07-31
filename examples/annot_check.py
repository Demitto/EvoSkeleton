import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# アノテーションファイルと画像ディレクトリのパス
annot_path = "h36m2Dpose/example_annot.npy"
image_dir = "h36m2Dpose/myimages"

# モデルの入力サイズ（学習時と同じサイズに揃える）
input_width = 288
input_height = 384

# アノテーションの読み込み
data = np.load(annot_path, allow_pickle=True).item()

# 最初のキーと対応する2Dキーポイント
first_key = list(data.keys())[0]
p2d = data[first_key]["p2d"]

# 画像読み込み
image_path = os.path.join(image_dir, first_key)
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"画像が見つかりません: {image_path}")

# BGR → RGB 変換
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 入力サイズにリサイズ（アノテーションと一致させる）
resized_image = cv2.resize(image, (input_width, input_height))

# プロット
plt.figure(figsize=(6, 8))
plt.imshow(resized_image)
plt.scatter(p2d[:, 0], p2d[:, 1], c='red', s=40, label="2D Keypoints")
for i, (x, y) in enumerate(p2d):
    plt.text(x + 3, y - 3, str(i), color='yellow', fontsize=8)

plt.title(f"Overlay: {first_key} (Resized to {input_width}x{input_height})")
plt.axis("off")
plt.legend()
plt.tight_layout()
plt.show()
