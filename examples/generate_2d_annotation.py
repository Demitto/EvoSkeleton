import os
import sys
sys.path.append("../")
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from libs.hhr.config import cfg, update_config
from libs.model.pose_hrnet import get_pose_net
from libs.hhr.utils.transforms import get_affine_transform
from libs.hhr.core.loss import get_max_preds_soft_pt

import argparse



# ====== 設定ファイルとモデル重み ======
cfg_file = './h36m2Dpose/cfgs.yaml'
model_path = './h36m2Dpose/final_state.pth'
img_dir = './h36m2Dpose/myimages'
output_npy = './h36m2Dpose/example_annot.npy'

update_config(cfg, argparse.Namespace(cfg=cfg_file))

model = get_pose_net(cfg, is_train=False)
model.load_state_dict(torch.load(model_path))
#model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

model = model.cuda()
#model = model.to(torch.device('cpu'))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToTensor(), normalize])

def xywh2cs(x, y, w, h, aspect_ratio=0.75):
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w / 200, h / 200], dtype=np.float32)
    return center, scale

def process_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    c, s = xywh2cs(0, 0, image.shape[1], image.shape[0])
    trans = get_affine_transform(c, s, 0, cfg.MODEL.IMAGE_SIZE)
    input_img = cv2.warpAffine(image, trans, tuple(cfg.MODEL.IMAGE_SIZE), flags=cv2.INTER_LINEAR)
    input_tensor = transform(input_img).unsqueeze(0).cuda()
    #input_tensor = transform(input_img).unsqueeze(0).to(torch.device('cpu'))
    return input_tensor, image

# ====== 推論して保存 ======
annot_dic = {}

for img_name in os.listdir(img_dir):
    if not img_name.endswith('.jpg'): continue
    input_tensor, _ = process_image(os.path.join(img_dir, img_name))
    with torch.no_grad():
        output = model(input_tensor)
        preds, _ = get_max_preds_soft_pt(output)
        pred = preds[0].cpu().numpy()  # shape: [17, 2]

    # ★ここを追加★ ---------------------------------
    re_order = [3,12,14,16,11,13,15,1,2,0,4,5,7,9,6,8,10]
    pred = pred[re_order]
    # ----------------------------------------------

    annot_dic[img_name] = {'p2d': pred}

np.save(output_npy, annot_dic)
print(f"Saved 2D annotations to {output_npy}")

