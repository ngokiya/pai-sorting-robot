# 視覚AIによる資源ゴミ自動仕分けロボット（Physical AI 応用1 最終課題）

MuJoCo 上の Franka Panda が、俯瞰 RGB-D カメラの画像から **自作 CNN で素材（PET / 缶 / 紙パック / ガラス）を判定** し、
素材ごとのビンへピック＆プレースで仕分けます。学習していない物体は **未知物体として検出し「手選別」ビンへ退避** させます。
GPU 不要（CPU 2 コアで学習〜動画生成まで完結）。

| 項目 | 内容 |
|---|---|
| シミュレータ | MuJoCo 3.x（Python）, ヘッドレス描画 (OSMesa / EGL) |
| ロボット | Franka Emika Panda（MuJoCo Menagerie）+ 吸着グリッパ（weld 拘束で模擬） |
| AI | RGB + 高さマップの 4ch 入力 小型 CNN（PyTorch, 約 2.8 万パラメータ）を合成データで学習。Mahalanobis 距離による未知物体検出 |
| 制御 | 減衰最小二乗 IK、5 次補間軌道、直線搬送、閉ループ再観測 |
| 結果 | 20 シード × 13 物体で把持成功 260/260、素材判定 94.2 %、既知素材同士の取り違え 0 |

詳細は `report/` の説明資料を参照してください。

## セットアップ

```bash
pip install -r requirements.txt
# Linux でディスプレイ無しの場合（GPU があれば egl、無ければ osmesa）
sudo apt install libosmesa6        # Ubuntu/Debian
export MUJOCO_GL=osmesa
```

Windows / macOS でディスプレイがある場合は `MUJOCO_GL` の設定は不要です。

## 実行手順

```bash
# 1. 合成データ生成（約 4 分, data/dataset.npz）
python src/gen_dataset.py --scenes 320

# 2. CNN 学習 + 未知物体検出器のフィッティング（約 90 秒, models/sorter_cnn.pt）
python src/train.py --epochs 30
#    → outputs/training_summary.png（学習曲線・混同行列・OOD 距離分布）, outputs/train_log.json

# 3. 仕分けタスク実行 + 動画生成（1280x720, 30fps, 等倍速。約 20〜40 分）
python src/run_task.py --seed 7 --n_per_class 3 --n_unknown 1 --out outputs/sorting_demo.mp4

# 4. 動画なしの高速評価（約 10 秒/シーン）
python src/run_task.py --seed 100 --n_per_class 3 --n_unknown 1 --no_video
#    → outputs/result_seed100.json
```

学習済みモデル `models/sorter_cnn.pt` を同梱しているので、手順 1–2 を省略して 3 から実行できます。

## ディレクトリ構成

```
src/scene.py        シーン生成（テーブル・ビン・物体ランダム生成・カメラ・画素↔ワールド変換）
src/perception.py   RGB-D 取得、前景抽出・連結成分、位置推定、クロップ生成
src/gen_dataset.py  合成データ生成
src/train.py        CNN 定義・学習・Mahalanobis 未知検出器・評価図
src/robot.py        数値 IK、軌道補間、吸着グリッパ、pick / place
src/run_task.py     閉ループ仕分けタスク、動画オーバーレイ、結果 JSON
assets/panda.xml    Menagerie の Panda に吸着ツール（サイト tcp）を追加したもの
assets/panda_meshes Panda のメッシュ（MuJoCo Menagerie, BSD-3-Clause）
models/             学習済み重み
report/             説明資料（PDF / HTML）と図
outputs/            学習・評価の出力（結果 JSON, 図）
```

## ライセンス

- 本リポジトリのコード: MIT License
- `assets/panda.xml`, `assets/panda_meshes/`: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause, `assets/LICENSE_panda_menagerie`)

## 参考

- AMP Robotics の AI 選別ロボット（80〜160 ピック/分）: https://www.recyclingproductnews.com/article/37333/
- シタラ興産の AI 選別ロボット導入事例（作業員 18 人→2 人）: https://newswitch.jp/p/11118
- K. Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018
