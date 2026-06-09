import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# --- 1. 基本設定與初始化 ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

# 檔案路徑與相機設定
MODEL_PATH = 'hand_landmarker.task'
OUTPUT_VIDEO_PATH = 'webcam_output.mp4'  # 儲存錄影結果的路徑
YOLO_MODEL_PATH = "yolo26n.pt" 
CAMERA_ID = 0  # 0 通常為筆電內建鏡頭，1 為外接鏡頭

# 初始化 YOLO 模型
print("載入 YOLO 模型中...")
yolo_model = YOLO(YOLO_MODEL_PATH)
TARGET_CLASSES = [42, 43, 44, 76]
class_names = {42: "Fork", 43: "Knife", 44: "Spoon", 76: "Scissors"}

# 設定 MediaPipe 參數
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=3,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. 開始辨識與處理 ---
with HandLandmarker.create_from_options(options) as landmarker:
    # 更改為讀取 Webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"錯誤：無法開啟攝影機 {CAMERA_ID}")
        exit()

    # 取得攝影機的預設解析度與 FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 60: fps = 30.0 # 確保 FPS 數值合理，避免寫入影片失敗

    # 設定影片寫入器 (可選：如果你想把 Webcam 畫面錄下來)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    print("開始即時畫面處理... (按 'q' 鍵退出)")
    
    # 記錄程式開始時間，用於計算 MediaPipe 需要的時間戳記
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝影機畫面")
            break

        # 針對 Webcam 即時影像，使用真實經過的時間來計算 timestamp_ms
        timestamp_ms = int((time.time() - start_time) * 1000)

        # --- A. YOLO 處理 (尋找餐具) ---
        yolo_results = yolo_model.predict(frame, conf=0.3, classes=TARGET_CLASSES, verbose=False)
        annotated_frame = yolo_results[0].plot() 
        
        detected_objects = []
        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0].item())
            detected_objects.append(class_names[class_id])

        # --- B. MediaPipe 處理 (找手) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 確保 timestamp 是嚴格遞增的
        mp_results = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_detected = False

        if mp_results.hand_landmarks:
            hand_detected = True
            for hand_idx, (handedness, landmarks) in enumerate(zip(
                mp_results.handedness, 
                mp_results.hand_landmarks
            )):
                mp_drawing.draw_landmarks(
                    annotated_frame,
                    landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # --- C. 判斷邏輯 ---
        if hand_detected and len(detected_objects) > 0:
            action_text = f"Action: Using {', '.join(set(detected_objects))}"
            cv2.putText(annotated_frame, action_text, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # 寫入錄影檔
        out.write(annotated_frame)
        
        # --- D. 即時顯示於視窗 ---
        cv2.imshow('Live Action Recognition', annotated_frame)

        # 監聽鍵盤事件，按下 'q' 鍵跳出迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("收到結束指令，準備關閉程式...")
            break

    # --- 3. 釋放資源 ---
    cap.release()
    out.release()
    cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
    print(f"錄影檔已成功儲存至：{OUTPUT_VIDEO_PATH}")