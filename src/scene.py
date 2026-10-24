"""
scene.py — MuJoCo シーン生成（テーブル・仕分けビン・資源ゴミ・カメラ）

カテゴリ（4クラス + 未知物体）:
  pet    : ペットボトル（細長い半透明シリンダ）
  can    : アルミ缶（短いメタリックなシリンダ）
  carton : 紙パック（直方体）
  glass  : ガラス瓶（太い不透明シリンダ）
  unknown: 学習時に見ていない形状（球・小さな立方体）→ 低信頼度で「手選別」へ
"""
import numpy as np
import os

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

CLASSES = ["pet", "can", "carton", "glass"]
CLASS_JA = {"pet": "ペットボトル", "can": "アルミ缶", "carton": "紙パック", "glass": "ガラス瓶", "unknown": "未知(手選別)"}

# ビン位置（ワールド座標, テーブル面 z=0）。BIN_AXIS はビンの長手方向（投入スロットをずらす向き）
BIN_POS = {
    "pet":     (0.25,  0.46),
    "can":     (0.56,  0.46),
    "carton":  (0.25, -0.46),
    "glass":   (0.56, -0.46),
    "unknown": (0.80,  0.00),
}
BIN_AXIS = {"pet": (1, 0), "can": (1, 0), "carton": (1, 0), "glass": (1, 0), "unknown": (0, 1)}
BIN_HALF = {"pet": (0.14, 0.09), "can": (0.14, 0.09), "carton": (0.14, 0.09), "glass": (0.14, 0.09), "unknown": (0.09, 0.14)}
SLOT_OFFSETS = (-0.07, 0.0, 0.07)   # 同じビンへ投入する順に使うスロット（物体が積み重ならないように）


def bin_slot(label, count):
    """label のビンに count 個目を投入するときの目標 (x, y)"""
    bx, by = BIN_POS[label]
    ax, ay = BIN_AXIS[label]
    off = SLOT_OFFSETS[count % len(SLOT_OFFSETS)]
    return (bx + ax * off, by + ay * off)
BIN_COLOR = {
    "pet":     "0.20 0.55 0.95 1",
    "can":     "0.75 0.75 0.78 1",
    "carton":  "0.80 0.55 0.25 1",
    "glass":   "0.20 0.60 0.30 1",
    "unknown": "0.55 0.55 0.55 1",
}

# 作業領域（物体をランダム配置する範囲）
WS_X = (0.40, 0.66)
WS_Y = (-0.24, 0.24)

# 俯瞰カメラ
TOPCAM_POS = (0.53, 0.0, 1.05)
TOPCAM_FOVY = 42.0
IMG_W, IMG_H = 480, 480


def _rand_color(rng, palette):
    base = np.array(palette[rng.integers(len(palette))], dtype=float)
    base[:3] = np.clip(base[:3] + rng.normal(0, 0.05, 3), 0, 1)
    return base


def sample_object(rng, cls):
    """カテゴリに応じて寸法・色をランダムサンプリング。半径/半高さ, rgba, 形状種別を返す。"""
    if cls == "pet":
        r = rng.uniform(0.022, 0.030)
        h = rng.uniform(0.045, 0.060)
        rgba = _rand_color(rng, [(0.75, 0.85, 1.0, 0.55), (0.85, 0.95, 0.9, 0.5), (0.95, 0.95, 0.95, 0.45),
                                 (0.70, 0.90, 0.75, 0.55)])
        return dict(type="cylinder", size=(r, h), rgba=rgba)
    if cls == "can":
        r = rng.uniform(0.028, 0.034)
        h = rng.uniform(0.028, 0.036)
        rgba = _rand_color(rng, [(0.85, 0.85, 0.88, 1), (0.85, 0.15, 0.15, 1), (0.15, 0.30, 0.80, 1),
                                 (0.90, 0.75, 0.20, 1), (0.20, 0.60, 0.30, 1)])
        return dict(type="cylinder", size=(r, h), rgba=rgba)
    if cls == "carton":
        a = rng.uniform(0.030, 0.038)
        h = rng.uniform(0.040, 0.055)
        rgba = _rand_color(rng, [(0.70, 0.50, 0.25, 1), (0.95, 0.95, 0.92, 1), (0.95, 0.85, 0.30, 1),
                                 (0.30, 0.45, 0.80, 1)])
        return dict(type="box", size=(a, a, h), rgba=rgba)
    if cls == "glass":
        r = rng.uniform(0.028, 0.036)
        h = rng.uniform(0.050, 0.065)
        rgba = _rand_color(rng, [(0.10, 0.35, 0.15, 0.95), (0.40, 0.22, 0.08, 0.95), (0.15, 0.15, 0.15, 0.95),
                                 (0.20, 0.60, 0.35, 0.9)])
        return dict(type="cylinder", size=(r, h), rgba=rgba)
    if cls == "unknown":
        if rng.random() < 0.5:
            r = rng.uniform(0.025, 0.035)
            rgba = _rand_color(rng, [(0.9, 0.4, 0.7, 1), (0.5, 0.1, 0.6, 1), (1.0, 0.5, 0.0, 1)])
            return dict(type="sphere", size=(r,), rgba=rgba)
        a = rng.uniform(0.018, 0.025)
        rgba = _rand_color(rng, [(0.9, 0.4, 0.7, 1), (0.1, 0.1, 0.1, 1), (0.0, 0.8, 0.8, 1)])
        return dict(type="box", size=(a, a * 2.2, a), rgba=rgba)
    raise ValueError(cls)


def sample_positions(rng, n, min_dist=0.10, max_tries=4000):
    """作業領域内に n 点を最小間隔 min_dist でランダム配置（入り切らなければ間隔を縮めて再試行）"""
    while True:
        pts = []
        for _ in range(max_tries):
            p = np.array([rng.uniform(*WS_X), rng.uniform(*WS_Y)])
            if all(np.linalg.norm(p - q) >= min_dist for q in pts):
                pts.append(p)
            if len(pts) == n:
                return pts
        min_dist *= 0.9


def object_xml(name, spec, pos, yaw):
    t = spec["type"]
    size = " ".join(f"{v:.4f}" for v in spec["size"])
    rgba = " ".join(f"{v:.3f}" for v in spec["rgba"])
    z = (spec["size"][1] if t == "cylinder" else spec["size"][-1] if t == "box" else spec["size"][0]) + 0.001
    quat = f"{np.cos(yaw/2):.5f} 0 0 {np.sin(yaw/2):.5f}"
    return f"""
    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {z:.4f}" quat="{quat}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_geom" type="{t}" size="{size}" rgba="{rgba}" mass="0.05"
            friction="1.0 0.02 0.001" condim="4"/>
    </body>"""


def bin_xml(name, cx, cy, color, half=(0.13, 0.085), height=0.06, wall=0.004):
    hx, hy = half
    g = f"""
    <body name="bin_{name}" pos="{cx:.3f} {cy:.3f} 0">
      <geom type="box" size="{hx} {hy} {wall}" pos="0 0 {wall}" rgba="{color}"/>
      <geom type="box" size="{wall} {hy} {height/2}" pos="{hx} 0 {height/2}" rgba="{color}"/>
      <geom type="box" size="{wall} {hy} {height/2}" pos="{-hx} 0 {height/2}" rgba="{color}"/>
      <geom type="box" size="{hx} {wall} {height/2}" pos="0 {hy} {height/2}" rgba="{color}"/>
      <geom type="box" size="{hx} {wall} {height/2}" pos="0 {-hy} {height/2}" rgba="{color}"/>
    </body>"""
    return g


def build_scene_xml(objects):
    """objects: list of dict(name, cls, spec, pos(x,y), yaw)"""
    obj_xml = "".join(object_xml(o["name"], o["spec"], o["pos"], o["yaw"]) for o in objects)
    bins = "".join(bin_xml(k, *BIN_POS[k], BIN_COLOR[k], half=BIN_HALF[k]) for k in BIN_POS)
    welds = "".join(
        f'\n    <weld name="suction_{o["name"]}" body1="hand" body2="{o["name"]}" active="false" solref="0.002 1"/>'
        for o in objects)
    tx, ty, tz = TOPCAM_POS
    xml = f"""
<mujoco model="pai_sorting">
  <include file="panda.xml"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <visual>
    <headlight diffuse="0.55 0.55 0.55" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <rgba haze="0.9 0.92 0.95 1"/>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.85 0.9 0.95" rgb2="0.6 0.7 0.8" width="256" height="1536"/>
    <texture type="2d" name="tabletex" builtin="flat" rgb1="0.93 0.93 0.90" width="64" height="64"/>
    <material name="tablemat" texture="tabletex" reflectance="0.05"/>
    <texture type="2d" name="floortex" builtin="checker" rgb1="0.35 0.37 0.40" rgb2="0.28 0.30 0.33" width="300" height="300" mark="edge" markrgb="0.5 0.5 0.5"/>
    <material name="floormat" texture="floortex" texrepeat="6 6" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0.3 0.3 2.0" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6" castshadow="true"/>
    <light pos="1.5 -1.0 1.5" dir="-0.6 0.4 -0.7" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.75" material="floormat"/>
    <!-- 作業テーブル（天板 z=0） -->
    <geom name="table" type="box" size="0.75 0.75 0.02" pos="0.35 0 -0.02" material="tablemat"/>
    <geom name="table_leg1" type="box" size="0.03 0.03 0.355" pos="-0.35 -0.7 -0.395" rgba="0.3 0.3 0.3 1"/>
    <geom name="table_leg2" type="box" size="0.03 0.03 0.355" pos="1.05 -0.7 -0.395" rgba="0.3 0.3 0.3 1"/>
    <geom name="table_leg3" type="box" size="0.03 0.03 0.355" pos="-0.35 0.7 -0.395" rgba="0.3 0.3 0.3 1"/>
    <geom name="table_leg4" type="box" size="0.03 0.03 0.355" pos="1.05 0.7 -0.395" rgba="0.3 0.3 0.3 1"/>
    <!-- 作業領域マーカー（ベルトコンベア風） -->
    <geom name="belt" type="box" size="{(WS_X[1]-WS_X[0])/2+0.05:.3f} {(WS_Y[1]-WS_Y[0])/2+0.05:.3f} 0.0015"
          pos="{(WS_X[0]+WS_X[1])/2:.3f} 0 0.0005" rgba="0.18 0.18 0.2 1"/>
    {bins}
    {obj_xml}
    <camera name="topcam" pos="{tx} {ty} {tz}" xyaxes="0 -1 0 1 0 0" fovy="{TOPCAM_FOVY}"/>
    <camera name="maincam" pos="1.75 -1.3 1.2" xyaxes="0.720 0.694 -0.000 -0.342 0.354 0.870" fovy="36"/>
    <camera name="sidecam" pos="0.55 1.6 0.7" xyaxes="-1 0 0 0 -0.4 0.92" fovy="45"/>
  </worldbody>
  <equality>{welds}
  </equality>
</mujoco>
"""
    return xml


def make_random_scene(rng, n_per_class=2, n_unknown=1):
    classes = []
    for c in CLASSES:
        classes += [c] * n_per_class
    classes += ["unknown"] * n_unknown
    rng.shuffle(classes)
    pts = sample_positions(rng, len(classes))
    objs = []
    for i, (c, p) in enumerate(zip(classes, pts)):
        objs.append(dict(name=f"obj{i}", cls=c, spec=sample_object(rng, c), pos=p, yaw=rng.uniform(0, np.pi)))
    return objs


def write_scene(objects, path=None):
    path = path or os.path.join(ASSET_DIR, "scene_generated.xml")
    with open(path, "w") as f:
        f.write(build_scene_xml(objects))
    return path


def pixel_to_world(u, v, z_plane=0.0, W=IMG_W, H=IMG_H):
    """俯瞰カメラのピクセル座標 (u,v) → テーブル平面上のワールド座標 (x,y)。
    ピンホールモデル: f = (H/2)/tan(fovy/2)。画像上=+x, 画像右=-y（xyaxes の設定による）"""
    f = (H / 2) / np.tan(np.deg2rad(TOPCAM_FOVY) / 2)
    depth = TOPCAM_POS[2] - z_plane
    xc = (u - W / 2) / f
    yc = (H / 2 - v) / f
    x = TOPCAM_POS[0] + yc * depth
    y = TOPCAM_POS[1] - xc * depth
    return x, y


def world_to_pixel(x, y, z_plane=0.0, W=IMG_W, H=IMG_H):
    f = (H / 2) / np.tan(np.deg2rad(TOPCAM_FOVY) / 2)
    depth = TOPCAM_POS[2] - z_plane
    yc = (x - TOPCAM_POS[0]) / depth
    xc = -(y - TOPCAM_POS[1]) / depth
    u = xc * f + W / 2
    v = H / 2 - yc * f
    return u, v
