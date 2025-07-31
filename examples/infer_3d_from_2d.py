import os, sys
sys.path.append("../")

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from libs.dataset.h36m.data_utils import unNormalizeData
import libs.visualization.viz as viz

# ---------- PATH ----------
ANN   = './h36m2Dpose/example_annot.npy'
MODEL = './h36m2Dpose/model.th'
STATS = './h36m2Dpose/stats.npy'
IMGS  = './h36m2Dpose/myimages'

# 2D順（あなたのnpy生成順と同じならそのまま）
re_order_indices = list(range(17))

def build_input_vector(p2d_raw, stats):
    dim_use_2d = np.array(stats['dim_use_2d'])
    mean_2d    = stats['mean_2d']
    std_2d     = stats['std_2d']

    p2d = p2d_raw[re_order_indices].reshape(-1).astype(np.float32)  # 34
    full = np.zeros_like(mean_2d, dtype=np.float32)                  # 64
    full[dim_use_2d] = p2d

    x_use = (full[dim_use_2d] - mean_2d[dim_use_2d]) / std_2d[dim_use_2d]
    return x_use[None, :]  # (1,32)

def run_cascade(cascade, x_np):
    x = torch.from_numpy(x_np.astype(np.float32))
    if torch.cuda.is_available(): x = x.cuda()
    with torch.no_grad():
        out = cascade[0](x)
        for m in cascade[1:]:
            out = out + m(x)
    return out.cpu().numpy()  # (1,48)

def restore_3d(pred48, stats):
    mean_3d, std_3d = stats['mean_3d'], stats['std_3d']
    dim_ignore_3d   = stats['dim_ignore_3d']
    out = unNormalizeData(pred48, mean_3d, std_3d, dim_ignore_3d)  # -> (1,96) か (1,48)
    return out.reshape(-1)

def main():
    data_dic = np.load(ANN,  allow_pickle=True).item()
    stats    = np.load(STATS, allow_pickle=True).item()

    cascade = torch.load(MODEL, weights_only=False)
    if torch.cuda.is_available():
        cascade = cascade.cuda()
    for m in cascade: m.eval()

    for i,(name,item) in enumerate(data_dic.items()):
        img = imageio.imread(os.path.join(IMGS, name))
        p2d = item['p2d']  # (17,2)

        x_in   = build_input_vector(p2d, stats)
        pred48 = run_cascade(cascade, x_in)
        pred3d = restore_3d(pred48, stats)  # ここでは**一切削らない**

        print(f"{i}: pred3d.size = {pred3d.size}")

        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(121)
        ax1.imshow(img); ax1.axis('off')

        ax2 = fig.add_subplot(122, projection='3d')
        # 公式可視化。内部で I,J を正しく使うので触らない
        viz.show3Dpose(pred3d, ax2, pred=True)

        plt.tight_layout()
        plt.show()

        if i>=16: break

if __name__ == "__main__":
    main()
