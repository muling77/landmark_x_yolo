import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from ultralytics import YOLO
from action_rules import get_action          # ← 新增：只 import 這一個函式

# =====================================================================
# 1. 路徑與常數設定
# =====================================================================
MODEL_PATH        = 'hand_landmarker.task'
OUTPUT_VIDEO_PATH = 'webcam_output.mp4'
YOLO_MODEL_PATH   = "yolo26n.pt"
CAMERA_ID         = 0

print("載入 YOLO 模型中...")
yolo_model    = YOLO(YOLO_MODEL_PATH)
TARGET_CLASSES = ['fork', 'knife', 'spoon', 'scissors']

# 判斷閾值（analyse_hands 用，與 action_rules.py 各自獨立）
OVERLAP_THRESH = 0.10

# =====================================================================
# 2. 幾何工具（僅保留 analyse_hands 需要的部分）
# =====================================================================

def hand_bbox(lm, W, H, pad=20):
    xs = [p.x*W for p in lm]; ys = [p.y*H for p in lm]
    return (max(0,int(min(xs))-pad), max(0,int(min(ys))-pad),
            min(W,int(max(xs))+pad), min(H,int(max(ys))+pad))


def iou(a, b) -> float:
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1) * max(0,iy2-iy1)
    if inter == 0: return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (union + 1e-6)


def wrist_to_obj_dist(lm, obj_cx, obj_cy, W, H) -> float:
    dx = lm[0].x*W - obj_cx; dy = lm[0].y*H - obj_cy
    return math.hypot(dx, dy) / math.hypot(W, H)

# =====================================================================
# 3. 核心判斷
#    ★ 改動點：is_grasping 改呼叫 get_action()，不再自己算 curl
# =====================================================================

def analyse_hands(all_lm, all_handedness, obj_box, cls_name, W, H):
    if not all_lm:
        return []

    ox1,oy1,ox2,oy2 = obj_box
    obj_cx = (ox1+ox2)/2; obj_cy = (oy1+oy2)/2

    dists = [wrist_to_obj_dist(lm, obj_cx, obj_cy, W, H) for lm in all_lm]
    nearest_idx = int(np.argmin(dists))

    results = []
    for idx, (lm, handedness) in enumerate(zip(all_lm, all_handedness)):
        is_nearest = (idx == nearest_idx)
        hand_score = float(handedness[0].score)

        # 最近那隻手才呼叫 get_action()，其餘手不做動作判定
        if is_nearest:
            action_name, debug_info = get_action(lm, obj_box, cls_name, W, H)
            is_grasping = (action_name != '')
        else:
            action_name = ''
            debug_info  = {}
            is_grasping = False

        results.append({
            'hand_idx'   : idx,
            'hand_label' : handedness[0].display_name,
            'is_nearest' : is_nearest,
            'is_grasping': is_grasping,
            'action_name': action_name,   # ★ 新增：直接帶出動作名稱
            'debug_info' : debug_info,    # ★ 新增：各規則的特徵數值
            'lm'         : lm,
        })
    return results

# =====================================================================
# 4. 繪圖工具
#    ★ 改動點：draw_hand 改顯示 debug_info（來自 action_rules）
# =====================================================================
mp_hands_conn     = mp.tasks.vision.HandLandmarksConnections
mp_drawing        = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

#手部資訊
def draw_hand(annotated, info, W, H):
    lm = info['lm']
    mp_drawing.draw_landmarks(
        annotated, lm,
        mp_hands_conn.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    wrist_px  = (int(lm[0].x*W), int(lm[0].y*H))
    tag_color = (0, 255, 128) if info['is_nearest'] else (180, 180, 180)
    mark = '[Target] ' if info['is_nearest'] else ''

    # 顯示來自 action_rules 的 debug 數值
    debug_str = ' '.join(f"{k}={v}" for k, v in info['debug_info'].items())
    label = f"{mark}{info['hand_label']} {debug_str}"
    cv2.putText(annotated, label,
                (wrist_px[0]-30, wrist_px[1]-18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, tag_color, 1)

#物件資訊
def draw_obj_box(annotated, box, cls_name, conf, grasping):
    x1,y1,x2,y2 = box
    color = (0, 220, 80) if grasping else (0, 165, 255)
    cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
    cv2.putText(annotated, f"{cls_name.title()} {conf:.2f}",
                (x1, max(y1-8,14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#頂部狀態列
def draw_status_bar(annotated, text, W):
    cv2.rectangle(annotated, (0,0), (W,44), (20,20,20), -1)
    cv2.putText(annotated, text, (12,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

# =====================================================================
# 5. 主程式（不變）
# =====================================================================
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

lm_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=3,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

yolo_names = yolo_model.names
target_ids = {cid for cid, name in yolo_names.items()
              if name.lower() in TARGET_CLASSES}


with HandLandmarker.create_from_options(lm_options) as landmarker:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"錯誤：無法開啟攝影機 {CAMERA_ID}"); exit()

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_vid = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (W, H)
    )

    frame_index = 0
    t = 0
    last_action = ''
    ACTION_THRESHOLD = 10      ### t的閾值 FIX HERE!!! ###
    print("開始，按 q 結束...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_index / fps) * 1000)
        frame_index += 1
        annotated = np.copy(frame)

        # ── A. MediaPipe ──────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        lm_result      = landmarker.detect_for_video(mp_img, timestamp_ms)
        all_lm         = lm_result.hand_landmarks
        all_handedness = lm_result.handedness

        # ── B. YOLO ───────────────────────────────────────────────────
        yolo_res = yolo_model(frame, classes=list(target_ids),
                              conf=0.3, verbose=False)
        
        best_conf      = 0.0
        best_hand_info = []

        for res in yolo_res:
            for box in res.boxes:
                cls_id   = int(box.cls[0])
                conf     = float(box.conf[0])
                cls_name = yolo_names[cls_id].lower()
                if cls_name not in TARGET_CLASSES: continue

                obj_box = tuple(map(int, box.xyxy[0]))

                # 傳入 cls_name，讓 analyse_hands 呼叫正確規則
                hand_info = analyse_hands(
                    all_lm, all_handedness, obj_box, cls_name, W, H)

                any_grasp = any(h['is_grasping'] for h in hand_info)
                draw_obj_box(annotated, obj_box, cls_name, conf, any_grasp)

                if conf > best_conf:
                    best_conf      = conf
                    best_hand_info = hand_info

        current_action = ''
        hand_conf = 0.0

        # 1. 從「最佳物件」中找出最靠近那隻手的動作與信心度
        if best_hand_info:
            for h in best_hand_info:
                if h['is_nearest'] and h['is_grasping']:
                    current_action = h['action_name']
                    # 這是上一步驟中加進 analyse_hands 的 MediaPipe 分數
                    hand_conf = h.get('hand_score', 0.0) 
                    break

        # 2. 核心計數邏輯 (融合 YOLO 與 MediaPipe)
        if current_action != '':
            YOLO_WEIGHT = 0.6
            HAND_WEIGHT = 0.4
            
            # 計算綜合分數
            composite_score = (best_conf * YOLO_WEIGHT) + (hand_conf * HAND_WEIGHT)
            #print("Yolo_conf=",best_conf)
            #print("Hand_conf=",hand_conf)
            print("conf=",composite_score)
            if current_action == last_action:
                if composite_score >= 0.25:
                    if t > 24 :
                        t = 24
                    else:    
                        t += 2
                elif 0.2 < composite_score < 0.25:
                    if t > 24 :
                        t = 24
                    else:
                        t += 1   
                else:
                    t -= 4   # 綜合分數過低，重度扣分
            else:
                # 動作改變，重新開始計數
                t = 1
                last_action = current_action
        else:
            # 沒偵測到任何抓握動作
            t -= 4

        # 確保計數器不會變成負數
        if t < 0:
            t = 0
            last_action = ''

        # 3. 根據計數器決定畫面要顯示的文字
        if t > ACTION_THRESHOLD:
            frame_action = last_action
            print('t=', t)
        else:
            frame_action = 'No grasping detected'
            print('t=', t)            

        # ── C. 畫骨架 ─────────────────────────────────────────────────
        if best_hand_info:
            for info in best_hand_info:
                draw_hand(annotated, info, W, H)
        elif all_lm:
            for lm, hdns in zip(all_lm, all_handedness):
                mp_drawing.draw_landmarks(
                    annotated, lm,
                    mp_hands_conn.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # ── D. 輸出 ───────────────────────────────────────────────────
        draw_status_bar(annotated, f"Action: {frame_action}", W)
        out_vid.write(annotated)
        cv2.imshow('Hand and Object Interaction Recognition', annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("結束..."); break

        if frame_index % 30 == 0:
            print(f"幀 {frame_index:5d} | {frame_action}")

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
    print(f"\n輸出影片：{OUTPUT_VIDEO_PATH}")