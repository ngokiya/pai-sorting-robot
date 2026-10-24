"""
perception.py — 俯瞰 RGB-D カメラによる物体検出（古典的セグメンテーション）と
                 CNN 分類用クロップの切り出し

処理の流れ:
  1. 俯瞰カメラから RGB 画像と深度画像を取得（MuJoCo の深度レンダリング = RGB-D センサ相当）
  2. 深度画像で「ベルト面より高いもの」を前景としてマスク（高さ閾値 1cm）
  3. 連結成分ラベリングで物体ごとに分離 → 重心ピクセル・上面高さを求める
  4. ピンホールモデルでピクセル座標をワールド座標へ変換（scene.pixel_to_world）
  5. 重心まわりを固定サイズ (CROP px) で切り出し、RGB + 高さマップの4chテンソルにする
"""
import numpy as np
import mujoco
from scipy import ndimage
import scene

CROP = 56          # CNN 入力の切り出しサイズ [px]
MIN_AREA = 60      # 物体とみなす最小ピクセル面積
HEIGHT_THRESH = 0.012  # ベルト面からの高さ閾値 [m]


class TopCamera:
    def __init__(self, model):
        self.m = model
        self.rgb = mujoco.Renderer(model, scene.IMG_H, scene.IMG_W)
        self.dep = mujoco.Renderer(model, scene.IMG_H, scene.IMG_W)
        self.dep.enable_depth_rendering()

    def capture(self, data):
        self.rgb.update_scene(data, camera="topcam")
        rgb = self.rgb.render().copy()
        self.dep.update_scene(data, camera="topcam")
        depth = self.dep.render().copy()           # カメラ光軸方向の距離 [m]
        height = scene.TOPCAM_POS[2] - depth       # テーブル面(z=0)からの高さマップ
        return rgb, height

    def close(self):
        self.rgb.close(); self.dep.close()


def detect(rgb, height, ws_margin=0.03):
    """前景セグメンテーション → 物体候補のリストを返す。
    各要素: dict(u, v, x, y, z_top, area, crop)"""
    mask = height > HEIGHT_THRESH
    # 作業領域(ベルト)外 = ビンや本体は除外
    H, W = mask.shape
    vv, uu = np.mgrid[0:H, 0:W]
    # 画素→ワールド (z=0 平面近似) で作業領域内かを判定
    xw, yw = scene.pixel_to_world(uu, vv, 0.0)
    inside = ((xw > scene.WS_X[0] - ws_margin) & (xw < scene.WS_X[1] + ws_margin) &
              (yw > scene.WS_Y[0] - ws_margin) & (yw < scene.WS_Y[1] + ws_margin))
    mask &= inside
    labels, n = ndimage.label(mask)
    dets = []
    for k in range(1, n + 1):
        region = labels == k
        area = int(region.sum())
        if area < MIN_AREA:
            continue
        # 上面高さ: 領域内の高さの上位分位（ノイズに頑健）
        z_top = float(np.percentile(height[region], 90))
        # 重心は「上面」のみで計算（側面が見えることによる透視歪みを抑える）
        top = region & (height > z_top - 0.012)
        v, u = ndimage.center_of_mass(top if top.sum() > 10 else region)
        x, y = scene.pixel_to_world(u, v, z_plane=z_top)
        crop = make_crop(rgb, height, u, v)
        dets.append(dict(u=float(u), v=float(v), x=float(x), y=float(y), z_top=z_top,
                         area=area, crop=crop, bbox=_bbox(region)))
    # 左上から順に（ロボット基部に近い順でも良い）
    dets.sort(key=lambda d: (d["x"], d["y"]))
    return dets


def _bbox(region):
    ys, xs = np.where(region)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def make_crop(rgb, height, u, v, size=CROP):
    """(u,v) を中心に size×size を切り出し、[RGB(0-1) ×3, 高さ(m)/0.1 ×1] の 4ch float 配列 (4,size,size) を返す。"""
    H, W = height.shape
    half = size // 2
    u0, v0 = int(round(u)) - half, int(round(v)) - half
    out = np.zeros((4, size, size), dtype=np.float32)
    uu0, vv0 = max(u0, 0), max(v0, 0)
    uu1, vv1 = min(u0 + size, W), min(v0 + size, H)
    out[0:3, vv0 - v0:vv1 - v0, uu0 - u0:uu1 - u0] = rgb[vv0:vv1, uu0:uu1].transpose(2, 0, 1) / 255.0
    out[3, vv0 - v0:vv1 - v0, uu0 - u0:uu1 - u0] = np.clip(height[vv0:vv1, uu0:uu1], 0, 0.3) / 0.1
    return out
