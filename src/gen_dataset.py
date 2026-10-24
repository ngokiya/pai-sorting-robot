"""
gen_dataset.py — シミュレータから CNN 学習用の合成データ（RGB-D クロップ）を生成

使い方:  MUJOCO_GL=osmesa python src/gen_dataset.py --scenes 300 --out data/dataset.npz
生成物:  X (N,4,56,56) float16, y (N,) int  [0:pet 1:can 2:carton 3:glass 4:unknown(評価専用)]
"""
import argparse, os, sys, time
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene, perception

LABEL = {c: i for i, c in enumerate(scene.CLASSES)}
LABEL["unknown"] = len(scene.CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="data/dataset.npz")
    ap.add_argument("--n_unknown", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    X, y, meta = [], [], []
    t0 = time.time()
    tmp_xml = os.path.join(scene.ASSET_DIR, "scene_dataset_tmp.xml")
    for s in range(a.scenes):
        objs = scene.make_random_scene(rng, n_per_class=2, n_unknown=a.n_unknown)
        path = scene.write_scene(objs, tmp_xml)
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        # アームを視野外（観測姿勢）に退避させた状態を qpos で直接作る
        d.qpos[:7] = OBSERVE_Q
        d.ctrl[:7] = OBSERVE_Q
        # 物体を少し落として静止させる（姿勢のばらつきも学習）
        for _ in range(200):
            mujoco.mj_step(m, d)
        cam = perception.TopCamera(m)
        rgb, hgt = cam.capture(d)
        dets = perception.detect(rgb, hgt)
        gt = {o["name"]: (d.xpos[m.body(o["name"]).id][:2].copy(), o["cls"]) for o in objs}
        for dt in dets:
            nm = min(gt, key=lambda k: np.hypot(*(gt[k][0] - [dt["x"], dt["y"]])))
            if np.hypot(*(gt[nm][0] - [dt["x"], dt["y"]])) > 0.03:
                continue
            X.append(dt["crop"].astype(np.float16)); y.append(LABEL[gt[nm][1]]); meta.append(s)
        cam.close()
        if (s + 1) % 50 == 0:
            print(f"{s+1}/{a.scenes} scenes, {len(X)} samples, {time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y), scene=np.array(meta))
    print("saved", a.out, np.bincount(y))


OBSERVE_Q = np.array([-0.011, -1.311, 0.003, -2.458, 0.003, 1.147, 0.776])  # TCP=(0.12,0,0.50) 俯瞰カメラ視野外

if __name__ == "__main__":
    main()
