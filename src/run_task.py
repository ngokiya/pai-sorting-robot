"""
run_task.py — 仕分けタスクの実行 + 動画生成（メイン）

使い方:
  MUJOCO_GL=osmesa python src/run_task.py --seed 7 --out outputs/sorting_demo.mp4
オプション:
  --no_video   動画を書き出さず結果JSONのみ（高速。精度検証用）
  --fps 30     動画フレームレート（シミュレーションは等倍）

閉ループ制御:
  観測姿勢へ退避 → RGB-D 取得 → 検出・CNN分類 → 1個目を把持 → ビンへ投入 → 観測姿勢へ… を物体が無くなるまで繰り返す
"""
import argparse, json, os, sys, time
import numpy as np
import mujoco
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene, robot, perception
from train import SorterCNN, predict, UNKNOWN, REJECT_THRESH

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
LABELS = scene.CLASSES + ["unknown"]
OBSERVE_Q = np.array([-0.011, -1.311, 0.003, -2.458, 0.003, 1.147, 0.776])
W, H = 1280, 720


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


class VideoOverlay:
    """メイン映像に俯瞰カメラの検出結果・推論結果・集計をオーバーレイする描画器"""
    def __init__(self):
        self.f_title = font(27, True); self.f_mid = font(21); self.f_small = font(17); self.f_big = font(44, True)
        self.inset = None          # 俯瞰カメラ画像 (PIL)
        self.phase = "起動"
        self.target = None         # dict(label, prob, probs, ja)
        self.tally = {k: [0, 0] for k in LABELS}  # label -> [投入数, うち正解数]
        self.n_detected = 0
        self.speed_note = "等倍速（リアルタイム）"
        self.t_sim = 0.0

    def set_observation(self, rgb, dets, labels, probs):
        im = Image.fromarray(rgb).convert("RGB")
        dr = ImageDraw.Draw(im)
        col = {"pet": (40, 140, 255), "can": (200, 200, 210), "carton": (230, 150, 60), "glass": (60, 180, 90), "unknown": (255, 60, 60)}
        for dt, lab, p in zip(dets, labels, probs):
            x0, y0, x1, y1 = dt["bbox"]
            c = col[LABELS[lab]]
            dr.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=c, width=3)
            txt = f"{LABELS[lab]} {p:.2f}"
            dr.rectangle([x0 - 2, y0 - 20, x0 + 8 * len(txt) + 4, y0 - 2], fill=c)
            dr.text((x0, y0 - 20), txt, fill=(0, 0, 0), font=font(15))
        self.inset = im.resize((300, 300))
        self.n_detected = len(dets)

    def draw(self, frame):
        im = Image.fromarray(frame).convert("RGBA")
        ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        # ヘッダ
        dr.rectangle([0, 0, W, 52], fill=(20, 24, 32, 200))
        dr.text((16, 10), "資源ゴミ自動仕分けロボット（視覚AI × Franka Panda, MuJoCo）", font=self.f_title, fill=(255, 255, 255, 255))
        dr.text((W - 330, 16), f"sim {self.t_sim:5.1f}s | {self.speed_note}", font=self.f_small, fill=(200, 200, 200, 255))
        # 右パネル: 俯瞰カメラ
        px, py = W - 320, 64
        dr.rectangle([px - 10, py - 10, W - 10, py + 300 + 10], fill=(20, 24, 32, 190))
        dr.text((px, py - 8 + 300 + 12 - 30), "", font=self.f_small)
        if self.inset is not None:
            ov.paste(self.inset.convert("RGBA"), (px, py))
        dr.text((px, py + 300 - 22), f"俯瞰 RGB-D カメラ  検出 {self.n_detected} 個", font=self.f_small, fill=(255, 255, 255, 255))
        # 右パネル: 推論結果
        qy = py + 330
        dr.rectangle([px - 10, qy - 10, W - 10, qy + 150], fill=(20, 24, 32, 190))
        dr.text((px, qy), f"▶ {self.phase}", font=self.f_small, fill=(255, 230, 120, 255))
        if self.target is not None:
            t = self.target
            dr.text((px, qy + 30), f"判定: {t['ja']}  (p={t['prob']:.2f})", font=self.f_mid, fill=(255, 255, 255, 255))
            for i, c in enumerate(scene.CLASSES):
                bx = px; by = qy + 64 + i * 20
                dr.text((bx, by), f"{c:7s}", font=self.f_small, fill=(200, 200, 200, 255))
                dr.rectangle([bx + 70, by + 4, bx + 70 + int(190 * t["probs"][i]), by + 16], fill=(90, 180, 255, 230))
                dr.text((bx + 265, by), f"{t['probs'][i]:.2f}", font=self.f_small, fill=(200, 200, 200, 255))
        # 左下: 集計
        ly = H - 150
        dr.rectangle([10, ly - 10, 330, H - 10], fill=(20, 24, 32, 190))
        dr.text((20, ly), "ビン投入数（正解/投入）", font=self.f_mid, fill=(255, 255, 255, 255))
        for i, k in enumerate(LABELS):
            n, ok = self.tally[k]
            dr.text((20, ly + 32 + i * 22), f"{scene.CLASS_JA[k]:9s} {ok}/{n}", font=self.f_small, fill=(230, 230, 230, 255))
        return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


def title_card(lines, sub=None):
    im = Image.new("RGB", (W, H), (22, 26, 34))
    dr = ImageDraw.Draw(im)
    y = 140
    for i, l in enumerate(lines):
        f = font(46, True) if i == 0 else font(28)
        dr.text((80, y), l, font=f, fill=(255, 255, 255))
        y += 70 if i == 0 else 44
    if sub:
        dr.text((80, H - 90), sub, font=font(20), fill=(170, 170, 170))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/sorting_demo.mp4")
    ap.add_argument("--model", default="models/sorter_cnn.pt")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--n_per_class", type=int, default=2)
    ap.add_argument("--n_unknown", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=14)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    objs = scene.make_random_scene(rng, a.n_per_class, a.n_unknown)
    xml = scene.write_scene(objs, os.path.join(scene.ASSET_DIR, f"scene_seed{a.seed}.xml"))
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[:7] = OBSERVE_Q; d.ctrl[:7] = OBSERVE_Q; d.ctrl[7] = 0
    mujoco.mj_forward(m, d)

    model = SorterCNN(); model.load_state_dict(torch.load(a.model, map_location="cpu")); model.eval()
    cam = perception.TopCamera(m)
    overlay = VideoOverlay()
    writer = None; main_r = None
    if not a.no_video:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "veryfast"])
        main_r = mujoco.Renderer(m, H, W)
        intro = title_card(["視覚AIによる資源ゴミ自動仕分けロボット",
                            "Physical AI 応用1 最終課題",
                            "・俯瞰 RGB-D カメラ → 自作 CNN で素材を分類（PET / 缶 / 紙パック / ガラス）",
                            "・学習していない物体は低信頼度として「手選別」ビンへ（未知物体の棄却）",
                            "・Franka Panda（吸着グリッパ）が数値IKでピック＆プレース",
                            "・シミュレータ: MuJoCo 3 / 動画は等倍速（リアルタイム）"],
                           f"環境: ランダム配置された {len(objs)} 個の物体（各素材 {a.n_per_class} 個 + 未知物体 {a.n_unknown} 個）")
        for _ in range(a.fps * 5):
            writer.append_data(intro)

    def frame_cb():
        if writer is None:
            return
        overlay.t_sim = ctl.sim_time
        main_r.update_scene(d, camera="maincam")
        writer.append_data(overlay.draw(main_r.render()))

    ctl = robot.PandaController(m, d, frame_cb=frame_cb, fps=a.fps)
    gt = {o["name"]: o["cls"] for o in objs}
    log = []
    failed = {}
    t_wall = time.time()
    overlay.phase = "観測"
    ctl.hold(0.5)
    for it in range(a.max_iters):
        # ---- 観測 ----
        overlay.phase = "観測（俯瞰カメラ）"
        ctl.move_joints(OBSERVE_Q, 0.8)
        ctl.hold(0.2)
        rgb, hgt = cam.capture(d)
        dets = perception.detect(rgb, hgt)
        if not dets:
            break
        X = torch.tensor(np.stack([dt["crop"] for dt in dets]))
        labels, probs, allp, dists = predict(model, X)
        overlay.set_observation(rgb, dets, labels, probs)
        overlay.phase = f"認識: {len(dets)} 個検出 → CNN で素材判定"
        ctl.hold(0.6)
        # ---- 対象選択（失敗が続いた位置は後回し）----
        order = sorted(range(len(dets)), key=lambda i: (failed.get(_key(dets[i]), 0), dets[i]["x"], dets[i]["y"]))
        i = order[0]
        dt, lab, p = dets[i], int(labels[i]), float(probs[i])
        label = LABELS[lab]
        overlay.target = dict(label=label, prob=p, probs=allp[i], ja=scene.CLASS_JA[label])
        # ---- 把持 ----
        overlay.phase = f"把持: {scene.CLASS_JA[label]} へ降下→吸着"
        held = ctl.pick((dt["x"], dt["y"]), dt["z_top"])
        if held is None:
            failed[_key(dt)] = failed.get(_key(dt), 0) + 1
            log.append(dict(iter=it, pred=label, prob=p, held=None, gt=None, correct=None))
            overlay.phase = "吸着失敗 → 再観測"
            continue
        true = gt[held]
        correct = (label == true)
        overlay.phase = f"搬送: {scene.CLASS_JA[label]} ビンへ投入"
        ctl.place(scene.bin_slot(label, overlay.tally[label][0]), obj_height=dt["z_top"])
        overlay.tally[label][0] += 1
        overlay.tally[label][1] += int(correct)
        log.append(dict(iter=it, pred=label, prob=p, held=held, gt=true, correct=bool(correct),
                        est_xy=[dt["x"], dt["y"]], z_top=dt["z_top"], sim_time=ctl.sim_time))
        print(f"[{it}] pred={label:7s} p={p:.2f} held={held} gt={true:7s} {'OK' if correct else 'NG'}  t={ctl.sim_time:.1f}s", flush=True)
        overlay.target = None
    overlay.phase = "完了"
    ctl.move_joints(OBSERVE_Q, 0.8)
    ctl.hold(1.0)

    # ---- 結果 ----
    picks = [l for l in log if l["held"]]
    n_ok = sum(l["correct"] for l in picks)
    bin_check = check_bins(m, d, objs)
    result = dict(seed=a.seed, n_objects=len(objs), n_picked=len(picks), n_correct=n_ok,
                  n_fail=len(log) - len(picks), sim_time=ctl.sim_time, wall_time=time.time() - t_wall,
                  in_correct_bin=bin_check, log=log, objects=[dict(name=o["name"], cls=o["cls"]) for o in objs])
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/result_seed{a.seed}.json", "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k not in ("log", "objects")}, ensure_ascii=False))
    if writer is not None:
        outro = title_card(["結果",
                            f"・物体 {len(objs)} 個中 {len(picks)} 個を搬送、分類正解 {n_ok} / {len(picks)}（吸着失敗 {len(log)-len(picks)} 回）",
                            f"・最終的に正しいビンに入った物体: {bin_check['correct']} / {len(objs)}",
                            f"・所要時間（シミュレーション時間）: {ctl.sim_time:.0f} 秒",
                            "・工夫: RGB-D 入力で形状(高さ)と色を同時に学習 / 未知物体の棄却 / 閉ループ再観測",
                            "・コード・レポートは GitHub リポジトリを参照"],
                           "MuJoCo 3.x / PyTorch / Franka Panda model: MuJoCo Menagerie (BSD-3)")
        for _ in range(a.fps * 6):
            writer.append_data(outro)
        writer.close()
        print("video saved:", a.out)
    cam.close()


def _key(dt):
    return (round(dt["x"], 2), round(dt["y"], 2))


def check_bins(m, d, objs):
    """最終状態で各物体がどのビンの上にあるかを判定（ワールド座標ベース）"""
    res = {"correct": 0, "wrong": 0, "none": 0, "detail": []}
    for o in objs:
        p = d.xpos[m.body(o["name"]).id]
        where = None
        for k, (bx, by) in scene.BIN_POS.items():
            hx, hy = scene.BIN_HALF[k]
            if abs(p[0] - bx) < hx + 0.02 and abs(p[1] - by) < hy + 0.02 and p[2] > 0.0:
                where = k
        res["detail"].append(dict(name=o["name"], cls=o["cls"], bin=where))
        if where is None:
            res["none"] += 1
        elif where == o["cls"]:
            res["correct"] += 1
        else:
            res["wrong"] += 1
    return res


if __name__ == "__main__":
    main()
