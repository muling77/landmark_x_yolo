"""
action_rules.py
---------------
動作判定規則模組。
所有幾何計算從 geometry.py import，這裡只負責「判定邏輯」。

新增動作：在下方寫新 class → 在 ACTION_REGISTRY 加一行。
換成 MLP：新增 MLPActionRule class，替換 ACTION_REGISTRY 對應項目。
tt.py 永遠不需要改動。
"""

from geometry import finger_curl, index_curl, thumb_index_dist, hand_bbox, iou


# =====================================================================
# 基底類別
# =====================================================================

class ActionRule:
    """
    所有動作規則的基底類別。
    子類別必須定義：
      target_object : str   對應的 YOLO 類別名稱（小寫）
      action_name   : str   成功時回傳的動作字串
      check()       : bool  判定是否符合此動作
      describe()    : dict  回傳各特徵數值（供畫面 debug 顯示）
    """
    target_object: str = ''
    action_name:   str = ''

    def check(self, lm, obj_box, W, H) -> bool:
        raise NotImplementedError

    def describe(self, lm, obj_box, W, H) -> dict:
        return {}


# =====================================================================
# 具體動作規則
# =====================================================================

class GenericGraspRule(ActionRule):
    def __init__(self, target_object, action_name, curl_thresh, overlap_thresh, pinch_min=0.0, index_straight_min=0.0):
        self.target_object = target_object
        self.action_name = action_name
        self.curl_thresh = curl_thresh  #手指彎曲程度的門檻值
        self.overlap_thresh = overlap_thresh   #手部與物件的重疊(IoU門檻值)
        self.pinch_min = pinch_min   #拇指與食指的最小距離
        self.index_straight_min = index_straight_min #食指伸直的最小角度

        self._cache_features = None # 加入暫存機制，避免同一幀重複計算

    def _features(self, lm, obj_box, W, H):
        if self._cache_features is not None:
            return self._cache_features
        
        # 1. 計算基礎通用特徵
        curl = finger_curl(lm, W, H)
        overlap = iou(hand_bbox(lm, W, H), obj_box)
        pinch = thumb_index_dist(lm, W, H)
        
        # 2. 效能優化：條件計算 (Lazy Evaluation)
        idx_curl = 0.0 # 給定一個預設值
        if self.index_straight_min > 0:
            # 只有像 knife 這種有設定 index_straight_min 的物品，才會進來算數學
            idx_curl = index_curl(lm, W, H)
        
        self._cache_features = curl, overlap, pinch, idx_curl
        return self._cache_features

    def check(self, lm, obj_box, W, H) -> bool:
        self._cache_features = None
        
        curl, overlap, pinch, idx_curl = self._features(lm, obj_box, W, H)
        
        # 3. 綜合多條件判定
        is_grasping = (curl < self.curl_thresh and 
                       overlap > self.overlap_thresh and 
                       pinch > self.pinch_min)
        
        # 額外動態條件：檢查食指伸直
        if self.index_straight_min > 0:
            is_grasping = is_grasping and (idx_curl > self.index_straight_min)
            
        return is_grasping

    def describe(self, lm, obj_box, W, H) -> dict:
        curl, overlap, pinch, idx_curl = self._features(lm, obj_box, W, H)
        desc = {'curl': f'{curl:.0f}', 'iou': f'{overlap:.2f}'}
        if self.pinch_min > 0:
             desc['pinch'] = f'{pinch:.2f}'
        if self.index_straight_min > 0:
             desc['idx_ang'] = f'{idx_curl:.0f}'
        return desc

# 註冊表變得極度簡潔，新增物件只需加一行參數
ACTION_REGISTRY = {
    'fork':     GenericGraspRule('fork', 'Grabbing Fork', 130, 0.01),
    'spoon':    GenericGraspRule('spoon', 'Grabbing Spoon', 130, 0.01, pinch_min=0.25),
    'scissors': GenericGraspRule('scissors', 'Grabbing Scissors', 115, 0.10),
    'knife':    GenericGraspRule('knife', 'Grabbing Knife', 170, 0.10, pinch_min=0.4, index_straight_min=140),
}

def get_action(lm, obj_box, cls_name: str, W: int, H: int) -> tuple[str, dict]:
    """
    tt.py 的唯一呼叫點。
    回傳 (action_name, debug_dict)。
    不符合任何規則時 action_name 為空字串 ''。
    """
    rule = ACTION_REGISTRY.get(cls_name)
    if rule is None:
        return '', {}
    if rule.check(lm, obj_box, W, H):
        return rule.action_name, rule.describe(lm, obj_box, W, H)
    return '', rule.describe(lm, obj_box, W, H)
