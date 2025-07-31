import numpy as np
import matplotlib.pyplot as plt

# ファイルパスを指定
npy_file = 'example_annot.npy'

# npyファイルを読み込み
try:
    data = np.load(npy_file, allow_pickle=True)
except Exception as e:
    print(f"ファイル読み込みエラー: {e}")
    exit()

print("データの型:", type(data))
print("データの形状:", data.shape)

# 中身を確認（最初の1サンプル）
if isinstance(data, np.ndarray):
    print("最初の要素の内容:")
    print(data[0])

    # 2D座標をプロット（関節数 × 2 形式が想定される）
    joints = data[0]
    if joints.ndim == 2 and joints.shape[1] == 2:
        plt.figure(figsize=(5, 7))
        plt.scatter(joints[:, 0], -joints[:, 1], c='r')  # Y軸反転して表示（画像っぽく）
        for i, (x, y) in enumerate(joints):
            plt.text(x, -y, str(i), fontsize=8)
        plt.title("example_annot.npy の最初のサンプル (2D keypoints)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    else:
        print("形式が想定と異なります。shape:", joints.shape)
else:
    print("読み込んだデータは ndarray ではありません。")
