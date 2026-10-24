# 隕冶ｦ哂I縺ｫ繧医ｋ雉・ｺ舌ざ繝溯・蜍穂ｻ募・縺代Ο繝懊ャ繝茨ｼ・hysical AI 蠢懃畑1 譛邨りｪｲ鬘鯉ｼ・
MuJoCo 荳翫・ Franka Panda 縺後∽ｿｯ迸ｰ RGB-D 繧ｫ繝｡繝ｩ縺ｮ逕ｻ蜒上°繧・**閾ｪ菴・CNN 縺ｧ邏譚撰ｼ・ET / 郛ｶ / 邏吶ヱ繝・け / 繧ｬ繝ｩ繧ｹ・峨ｒ蛻､螳・* 縺励・邏譚舌＃縺ｨ縺ｮ繝薙Φ縺ｸ繝斐ャ繧ｯ・・・繝ｬ繝ｼ繧ｹ縺ｧ莉募・縺代∪縺吶ょｭｦ鄙偵＠縺ｦ縺・↑縺・黄菴薙・ **譛ｪ遏･迚ｩ菴薙→縺励※讀懷・縺励梧焔驕ｸ蛻･縲阪ン繝ｳ縺ｸ騾驕ｿ** 縺輔○縺ｾ縺吶・GPU 荳崎ｦ・ｼ・PU 2 繧ｳ繧｢縺ｧ蟄ｦ鄙偵懷虚逕ｻ逕滓・縺ｾ縺ｧ螳檎ｵ撰ｼ峨・
| 鬆・岼 | 蜀・ｮｹ |
|---|---|
| 繧ｷ繝溘Η繝ｬ繝ｼ繧ｿ | MuJoCo 3.x・・ython・・ 繝倥ャ繝峨Ξ繧ｹ謠冗判 (OSMesa / EGL) |
| 繝ｭ繝懊ャ繝・| Franka Emika Panda・・uJoCo Menagerie・・ 蜷ｸ逹繧ｰ繝ｪ繝・ヱ・・eld 諡俶據縺ｧ讓｡謫ｬ・・|
| AI | RGB + 鬮倥＆繝槭ャ繝励・ 4ch 蜈･蜉・蟆丞梛 CNN・・yTorch, 邏・2.8 荳・ヱ繝ｩ繝｡繝ｼ繧ｿ・峨ｒ蜷域・繝・・繧ｿ縺ｧ蟄ｦ鄙偵・ahalanobis 霍晞屬縺ｫ繧医ｋ譛ｪ遏･迚ｩ菴捺､懷・ |
| 蛻ｶ蠕｡ | 貂幄｡ｰ譛蟆丈ｺ御ｹ・IK縲・ 谺｡陬憺俣霆碁％縲∫峩邱壽成騾√・哩繝ｫ繝ｼ繝怜・隕ｳ貂ｬ |
| 邨先棡 | 20 繧ｷ繝ｼ繝・ﾃ・13 迚ｩ菴薙〒謚頑戟謌仙粥 260/260縲∫ｴ譚仙愛螳・94.2 %縲∵里遏･邏譚仙酔螢ｫ縺ｮ蜿悶ｊ驕輔∴ 0 |

隧ｳ邏ｰ縺ｯ `report/` 縺ｮ隱ｬ譏手ｳ・侭繧貞盾辣ｧ縺励※縺上□縺輔＞縲・
## 繧ｻ繝・ヨ繧｢繝・・

```bash
pip install -r requirements.txt
# Linux 縺ｧ繝・ぅ繧ｹ繝励Ξ繧､辟｡縺励・蝣ｴ蜷茨ｼ・PU 縺後≠繧後・ egl縲∫┌縺代ｌ縺ｰ osmesa・・sudo apt install libosmesa6        # Ubuntu/Debian
export MUJOCO_GL=osmesa
```

Windows / macOS 縺ｧ繝・ぅ繧ｹ繝励Ξ繧､縺後≠繧句ｴ蜷医・ `MUJOCO_GL` 縺ｮ險ｭ螳壹・荳崎ｦ√〒縺吶・
## 螳溯｡梧焔鬆・
```bash
# 1. 蜷域・繝・・繧ｿ逕滓・・育ｴ・4 蛻・ data/dataset.npz・・python src/gen_dataset.py --scenes 320

# 2. CNN 蟄ｦ鄙・+ 譛ｪ遏･迚ｩ菴捺､懷・蝎ｨ縺ｮ繝輔ぅ繝・ユ繧｣繝ｳ繧ｰ・育ｴ・90 遘・ models/sorter_cnn.pt・・python src/train.py --epochs 30
#    竊・outputs/training_summary.png・亥ｭｦ鄙呈峇邱壹・豺ｷ蜷瑚｡悟・繝ｻOOD 霍晞屬蛻・ｸ・ｼ・ outputs/train_log.json

# 3. 莉募・縺代ち繧ｹ繧ｯ螳溯｡・+ 蜍慕判逕滓・・・280x720, 30fps, 遲牙埼溘らｴ・20縲・0 蛻・ｼ・python src/run_task.py --seed 7 --n_per_class 3 --n_unknown 1 --out outputs/sorting_demo.mp4

# 4. 蜍慕判縺ｪ縺励・鬮倬溯ｩ穂ｾ｡・育ｴ・10 遘・繧ｷ繝ｼ繝ｳ・・python src/run_task.py --seed 100 --n_per_class 3 --n_unknown 1 --no_video
#    竊・outputs/result_seed100.json
```

蟄ｦ鄙呈ｸ医∩繝｢繝・Ν `models/sorter_cnn.pt` 繧貞酔譴ｱ縺励※縺・ｋ縺ｮ縺ｧ縲∵焔鬆・1窶・ 繧堤怐逡･縺励※ 3 縺九ｉ螳溯｡後〒縺阪∪縺吶・
## 繝・ぅ繝ｬ繧ｯ繝医Μ讒区・

```
src/scene.py        繧ｷ繝ｼ繝ｳ逕滓・・医ユ繝ｼ繝悶Ν繝ｻ繝薙Φ繝ｻ迚ｩ菴薙Λ繝ｳ繝繝逕滓・繝ｻ繧ｫ繝｡繝ｩ繝ｻ逕ｻ邏竊斐Ρ繝ｼ繝ｫ繝牙､画鋤・・src/perception.py   RGB-D 蜿門ｾ励∝燕譎ｯ謚ｽ蜃ｺ繝ｻ騾｣邨先・蛻・∽ｽ咲ｽｮ謗ｨ螳壹√け繝ｭ繝・・逕滓・
src/gen_dataset.py  蜷域・繝・・繧ｿ逕滓・
src/train.py        CNN 螳夂ｾｩ繝ｻ蟄ｦ鄙偵・Mahalanobis 譛ｪ遏･讀懷・蝎ｨ繝ｻ隧穂ｾ｡蝗ｳ
src/robot.py        謨ｰ蛟､ IK縲∬ｻ碁％陬憺俣縲∝精逹繧ｰ繝ｪ繝・ヱ縲｝ick / place
src/run_task.py     髢峨Ν繝ｼ繝嶺ｻ募・縺代ち繧ｹ繧ｯ縲∝虚逕ｻ繧ｪ繝ｼ繝舌・繝ｬ繧､縲∫ｵ先棡 JSON
assets/panda.xml    Menagerie 縺ｮ Panda 縺ｫ蜷ｸ逹繝・・繝ｫ・医し繧､繝・tcp・峨ｒ霑ｽ蜉縺励◆繧ゅ・
assets/panda_meshes Panda 縺ｮ繝｡繝・す繝･・・uJoCo Menagerie, BSD-3-Clause・・models/             蟄ｦ鄙呈ｸ医∩驥阪∩
report/             隱ｬ譏手ｳ・侭・・DF / HTML・峨→蝗ｳ
outputs/            蟄ｦ鄙偵・隧穂ｾ｡縺ｮ蜃ｺ蜉幢ｼ育ｵ先棡 JSON, 蝗ｳ・・```

## 繝ｩ繧､繧ｻ繝ｳ繧ｹ

- 譛ｬ繝ｪ繝昴ず繝医Μ縺ｮ繧ｳ繝ｼ繝・ MIT License
- `assets/panda.xml`, `assets/panda_meshes/`: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (BSD-3-Clause, `assets/LICENSE_panda_menagerie`)

## 蜿り・
- AMP Robotics 縺ｮ AI 驕ｸ蛻･繝ｭ繝懊ャ繝茨ｼ・0縲・60 繝斐ャ繧ｯ/蛻・ｼ・ https://www.recyclingproductnews.com/article/37333/
- 繧ｷ繧ｿ繝ｩ闊育肇縺ｮ AI 驕ｸ蛻･繝ｭ繝懊ャ繝亥ｰ主・莠倶ｾ具ｼ井ｽ懈･ｭ蜩｡ 18 莠ｺ竊・ 莠ｺ・・ https://newswitch.jp/p/11118
- K. Lee et al., "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018

