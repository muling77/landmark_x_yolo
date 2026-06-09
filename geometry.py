"""
geometry.py
-----------
所有幾何計算工具。
純數學函式，不含任何判定邏輯與閾值。
action_rules.py 和 tt.py 都從這裡 import，不重複定義。

之後加入 MLP：
  - 這裡的函式仍然需要，用來把 landmark 轉成 feature vector
  - 判定邏輯的替換發生在 action_rules.py，不在這裡
"""

import math
import numpy as np


# =====================================================================
# 向量與角度
# =====================================================================

def vec_angle(a, b, c) -> float:
    """
    b 為頂點，a→b→c 的夾角（度）。
    輸入：(x, y) tuple。
    """
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# =====================================================================
# 手部特徵計算
# =====================================================================

def finger_curl(lm, W, H) -> float:
    """
    四根手指（食～小）MCP-PIP-TIP 夾角平均值。
    角度越小 → 手指越彎 → 越像握拳。
    """
    joints = [(5,6,8), (9,10,12), (13,14,16), (17,18,20)]
    angles = []
    for mcp, pip, tip in joints:
        a = (lm[mcp].x*W, lm[mcp].y*H)
        b = (lm[pip].x*W, lm[pip].y*H)
        c = (lm[tip].x*W, lm[tip].y*H)
        angles.append(vec_angle(a, b, c))
    return float(np.mean(angles))


def index_curl(lm, W, H) -> float:
    """食指單獨的彎曲角度（MCP-PIP-TIP）。"""
    a = (lm[5].x*W, lm[5].y*H)
    b = (lm[6].x*W, lm[6].y*H)
    c = (lm[8].x*W, lm[8].y*H)
    return vec_angle(a, b, c)


def thumb_index_dist(lm, W, H) -> float:
    """
    拇指尖(4)到食指尖(8)的距離，以手掌寬度正規化。
    數值越大 → 越展開（握刀姿勢）；越小 → 越捏合（捏取姿勢）。
    """
    tx, ty = lm[4].x*W, lm[4].y*H
    ix, iy = lm[8].x*W, lm[8].y*H
    palm_w  = math.hypot((lm[5].x-lm[17].x)*W,
                          (lm[5].y-lm[17].y)*H) + 1e-6
    return math.hypot(tx-ix, ty-iy) / palm_w


def palm_center(lm, W, H) -> tuple[float, float]:
    """
    手掌中心座標（像素）：取手腕(0)與中指根(9)的中點。
    比單用手腕更接近手掌幾何中心。
    """
    cx = (lm[0].x + lm[9].x) / 2 * W
    cy = (lm[0].y + lm[9].y) / 2 * H
    return cx, cy


# =====================================================================
# Bounding box 工具
# =====================================================================

def hand_bbox(lm, W, H, pad=20) -> tuple[int,int,int,int]:
    """
    21 個 landmark 的外接矩形（含 padding）。
    回傳 (x1, y1, x2, y2)，像素座標。
    """
    xs = [p.x*W for p in lm]
    ys = [p.y*H for p in lm]
    return (max(0,   int(min(xs))-pad),
            max(0,   int(min(ys))-pad),
            min(W,   int(max(xs))+pad),
            min(H,   int(max(ys))+pad))


def iou(a, b) -> float:
    """
    兩個 (x1,y1,x2,y2) bounding box 的 IoU。
    用於判斷手部與物件的重疊程度。
    """
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-6)


# =====================================================================
# 距離計算
# =====================================================================

def wrist_to_obj_dist(lm, obj_cx, obj_cy, W, H) -> float:
    """
    手腕(點0)到物件中心的距離，以畫面對角線正規化。
    用於判斷哪隻手離物件最近。
    """
    dx = lm[0].x*W - obj_cx
    dy = lm[0].y*H - obj_cy
    return math.hypot(dx, dy) / math.hypot(W, H)


# =====================================================================
# Feature vector（為 MLP 準備，現階段也可用於 debug）
# =====================================================================

def to_feature_vector(lm, obj_box, W, H) -> np.ndarray:
    """
    將單隻手的 landmark + 物件 box 轉成 1D feature vector。
    現階段不使用，MLP 訓練/推論時直接呼叫。

    輸出維度：
      63  = 21點 × (x, y, z)          ← 正規化座標，直接取值
       4  = 手部特徵 (curl, index_curl, thumb_index_dist, iou)
       2  = 物件中心相對手腕的偏移 (dx, dy)，正規化
    共 69 維。
    """
    # 21點展平
    coords = []
    for p in lm:
        coords += [p.x, p.y, p.z]

    # 手部幾何特徵
    ox1,oy1,ox2,oy2 = obj_box
    obj_cx = (ox1+ox2)/2;  obj_cy = (oy1+oy2)/2
    hbox   = hand_bbox(lm, W, H)
    feats  = [
        finger_curl(lm, W, H)      / 180.0,   # 正規化到 [0,1]
        index_curl(lm, W, H)       / 180.0,
        thumb_index_dist(lm, W, H) / 3.0,      # 通常不超過 3 倍手掌寬
        iou(hbox, obj_box),
        (lm[0].x*W - obj_cx) / W,              # 手腕到物件 x 偏移
        (lm[0].y*H - obj_cy) / H,              # 手腕到物件 y 偏移
    ]

    return np.array(coords + feats, dtype=np.float32)
