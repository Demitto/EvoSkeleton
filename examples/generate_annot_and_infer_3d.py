#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
infer_and_show.py
    – myimages 配下の jpg を一括で 2D→3D 推論して表示するスクリプト
"""

import os, sys, cv2, argparse
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ---------- ライブラリの import パス ----------
sys.path.append("../")           # libs/ が 1 つ上の階層にある想定
from libs.hhr.config import cfg, update_config
from libs.model.pose_hrnet import get_pose_net
from libs.hhr.utils.transforms import get_affine_transform
from libs.hhr.core.loss import get_max_preds_soft_pt
from libs.dataset.h36m.data_utils import unNormalizeData
import libs.visualization.viz as viz          # 3D 可視化

# ---------- 引数 ----------
parser = argparse.ArgumentParser()
parser.add_argument("--cfg",   default="./h36m2Dpose/cfgs.yaml")
parser.add_argument("--hrnet", default="./h36m2Dpose/final_state.pth")
parser.add_argument("--casc",  default="./h36m2Dpose/model.th")
parser.add_argument("--stats", default="./h36m2Dpose/stats.npy")
parser.add_argument("--imgs",  default="./h36m2Dpose/myimages")
args = parser.parse_args()

# ---------- HRNet 2D モデル ----------
update_config(cfg, argparse.Namespace(cfg=args.cfg))
hrnet = get_pose_net(cfg, is_train=False)
hrnet.load_state_dict(torch.load(args.hrnet, map_location="cpu"))
hrnet.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hrnet = hrnet.to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
to_tensor = transforms.Compose([transforms.ToTensor(), normalize])

# ---------- 3D Cascade ----------
cascade = torch.load(args.casc, weights_only=False, map_location="cpu")
for m in cascade: m.eval()
cascade = cascade.to(device)

stats = np.load(args.stats, allow_pickle=True).item()
dim_use_2d     = np.array(stats["dim_use_2d"])
mean_2d, std_2d = stats["mean_2d"], stats["std_2d"]
mean_3d, std_3d = stats["mean_3d"], stats["std_3d"]
dim_ignore_3d   = stats["dim_ignore_3d"]

# ---------- 2D→3D 関数 ----------
# 2D joint 並び替え（HRNet → Cascade 用）
re_order = [3,12,14,16,11,13,15,1,2,0,4,5,7,9,6,8,10]  # ★必要に応じて変更
re_order_indices = list(range(17))                      # 3D 側はこの順で受け取る前提

def xywh2cs(x, y, w, h, aspect_ratio=0.75):
    c = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    s = np.array([w / 200, h / 200], dtype=np.float32)
    return c, s

def process_image(im_path):
    img_bgr = cv2.imread(im_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    c, s = xywh2cs(0, 0, img_rgb.shape[1], img_rgb.shape[0])
    trans  = get_affine_transform(c, s, 0, cfg.MODEL.IMAGE_SIZE)
    inp    = cv2.warpAffine(img_rgb, trans,
                            tuple(cfg.MODEL.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    inp_t  = to_tensor(inp).unsqueeze(0).to(device)
    return inp_t, img_rgb, trans

def pred_2d(inp_t):
    """HRNet の出力を (17,2) 座標と (17,) confidence で返す"""
    with torch.no_grad():
        heat = hrnet(inp_t)                         # (1,17,H,W)
        coords, conf = get_max_preds_soft_pt(heat)  # ← 2 値取得
    return coords[0].cpu().numpy(), conf[0].cpu().numpy().flatten()


def warp_back(points, trans):
    """256x192 空間の点を元画像座標へ戻す"""
    inv = cv2.invertAffineTransform(trans)    # (2,3)
    pts_h = np.concatenate([points, np.ones([points.shape[0],1])], axis=1)
    return (inv @ pts_h.T).T                  # (N,2)

def build_x_use(p2d_raw):
    p2d = p2d_raw[re_order_indices].reshape(-1).astype(np.float32)   # 34
    full = np.zeros_like(mean_2d, dtype=np.float32)                  # 64
    full[dim_use_2d] = p2d
    x_use = (full[dim_use_2d] - mean_2d[dim_use_2d]) / std_2d[dim_use_2d]
    return x_use.astype(np.float32)[None, :]

def run_cascade(x_np):
    x = torch.from_numpy(x_np).to(device).float()
    with torch.no_grad():
        out = cascade[0](x)
        for m in cascade[1:]:
            out = out + m(x)
    return out.cpu().numpy()        # (1,48)

def restore_3d(pred48):
    return unNormalizeData(pred48, mean_3d, std_3d,
                           dim_ignore_3d).reshape(-1)  # (96,) or (48,)

# ---------- メインループ ----------
jpgs = sorted([f for f in os.listdir(args.imgs) if f.lower().endswith(".jpg")])
if not jpgs:
    raise RuntimeError(f"No .jpg found in {args.imgs}")

for idx, fname in enumerate(jpgs):
    path = os.path.join(args.imgs, fname)
    inp_t, img_rgb, trans = process_image(path)

    # ---------- 2D 推論 ----------
    p2d_raw, conf = pred_2d(inp_t)     # ← 修正
    p2d      = p2d_raw[re_order]
    conf     = conf[re_order]
    p2d_img  = warp_back(p2d, trans)

    # ---------- 信頼度しきい値フィルタ（任意）
    #invalid = conf < 0.15
    #p2d[invalid] = np.nan              # これで build_x_use 内で 0 になる

    # ---------- 描画 ----------
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(121)
    ax1.imshow(img_rgb); ax1.axis("off")
    ax1.set_title("2D keypoints with confidence")

    colors = plt.cm.viridis(np.clip(conf,0,1))
    ax1.scatter(p2d_img[:,0], p2d_img[:,1], s=40,
                c=colors, marker='o', edgecolors='k')
    for (x,y),c in zip(p2d_img, conf):
        ax1.text(x, y, f"{c:.2f}", color="white",
                fontsize=8, ha='center', va='center')

    # コンソールに信頼度を出力
    print(f"\n[{idx+1}/{len(jpgs)}] {fname}  confidence per joint:")
    for j,c in enumerate(conf):
        print(f"  {j:02d}: {c:.3f}")

    # ---------- 3D 推論 ----------
    x_use   = build_x_use(p2d)              # (1,32)
    pred48  = run_cascade(x_use)
    pred3d  = restore_3d(pred48)

    # ---------- 描画 ----------
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_rgb); ax1.axis("off")
    ax1.scatter(p2d_img[:,0], p2d_img[:,1], s=20, marker='o')  # 2D キーポイント
    for pt in p2d_img:
        ax1.text(pt[0], pt[1], '.', color='yellow', fontsize=14, ha='center')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    viz.show3Dpose(pred3d, ax2, pred=True)  # Skeleton 可視化
    ax2.set_title("3D pose")

    plt.suptitle(f"[{idx+1}/{len(jpgs)}]  {fname}")
    plt.tight_layout()
    plt.show()

print("=== 完了 ===")
