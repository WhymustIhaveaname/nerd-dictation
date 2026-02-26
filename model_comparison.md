# Model Comparison – 2026-02-24

## Setup

- Hardware: NVIDIA RTX 4060, running on CPU (CUDA fallback due to missing libcurand)
- Test corpus: 8 recordings (4 categories × 2 speeds), see `tests/test_wavs/manifest.json`
  - **zh**: 纯中文（函数描述）
  - **en**: 纯英文（function description）
  - **zh_en_lite**: 中文 + 少量英文术语（transformer, epoch）
  - **zh_en_heavy**: 中文 + 大量英文术语（backend, python, fastapi, postgresql, sqlalchemy, orm）
- Metric: Character Error Rate (CER), lower is better
- Script: `tests/test_model_comparison.py`

## CER Validation

Before trusting the metric, we manually verified CER against human perception on several samples:

| Sample | Model | CER | Observation |
|--------|-------|-----|-------------|
| zh_slow | vosk-large | 2% | 35 字只错 1 字（是→时），几乎完美 |
| zh_slow | sherpa-small | 9% | 错 3 处（余→鱼, 浮→辅, 值是→执），大意可读 |
| zh_en_lite_slow | sherpa-small | 35% | 中文全对，英文全错（transformer→穿行白） |
| zh_en_heavy_slow | sherpa-small | 60% | 结构对一半，英文词大部分错 |
| zh_en_heavy_normal | vosk-small | 97% | 50 字只识别出"我们啊"，几乎全废 |

**Conclusion**: CER accurately reflects perceived recognition quality.
2-9% = near perfect, 35% = Chinese correct but English garbled,
60% = roughly half correct, 97-100% = nearly nothing recognized.

## Results

```
                          vosk-small        vosk-large       sherpa-small      sherpa-large
wav                       CER    time      CER    time      CER    time      CER    time
------------------------------------------------------------------------------------------
zh_slow.wav               32%    5.1s       2%    6.1s       9%    0.5s       9%    1.0s
zh_normal.wav             64%    6.0s       7%    8.8s      11%    0.4s       7%    0.9s
en_slow.wav              100%    4.5s     100%    8.6s      52%    0.4s      47%    0.8s
en_normal.wav            100%    1.3s     100%    4.8s     100%    0.2s      93%    0.4s
zh_en_lite_slow.wav       73%    5.9s      44%    8.2s      35%    0.4s      15%    0.9s
zh_en_lite_normal.wav     96%    2.2s      67%    7.5s      42%    0.3s      40%    0.5s
zh_en_heavy_slow.wav      85%    5.9s      79%    8.9s      60%    0.4s      49%    0.9s
zh_en_heavy_normal.wav    97%    3.0s      96%    8.3s      63%    0.3s      54%    0.7s
------------------------------------------------------------------------------------------
AVG CER (lower=better)    81%              62%              46%              39%
TOT time                       33.9s             61.2s             2.9s             6.1s
```

## Analysis

- **Overall**: sherpa-large (39%) > sherpa-small (46%) > vosk-large (62%) > vosk-small (81%)
- **Pure Chinese**: vosk-large 和 sherpa-large 并列最优 (2%/7% vs 9%/7%)
- **Pure English**: 均较差, sherpa-large 略好 (47%/93%), 首次在正常语速下有输出
- **Mixed zh-en**: sherpa-large 明显领先, 尤其 lite 慢速 (15% vs 35%), heavy 慢速 (49% vs 60%)
- **Speed**: sherpa-small (2.9s) 最快, sherpa-large (6.1s) 次之, 均远快于 vosk (34-61s)

## Noise Reduction Effect (sherpa-large)

Using `--noise-reduction` with `noisereduce` library. Script: `tests/test_noise_effect.py`

```
                               lv0            lv1            lv2
wav                       CER   time     CER   time     CER   time
--------------------------------------------------------------------
zh_slow.wav                9%   1.0s      5%   1.7s     18%   1.1s
zh_normal.wav              7%   1.0s      9%   1.0s     20%   1.0s
en_slow.wav               47%   0.8s     42%   0.9s     75%   0.9s
en_normal.wav             93%   0.4s     93%   0.5s    100%   0.5s
zh_en_lite_slow.wav       15%   0.9s     17%   1.0s     27%   1.0s
zh_en_lite_normal.wav     40%   0.7s     40%   0.8s     85%   0.6s
zh_en_heavy_slow.wav      49%   1.0s     46%   1.1s     70%   1.0s
zh_en_heavy_normal.wav    54%   0.8s     58%   0.8s     87%   0.7s
--------------------------------------------------------------------
AVG CER / TOT time        39%   6.5s     39%   7.7s     60%   6.8s
```

- **Level 1** (light): 平均 CER 持平 (39%), 个别样本小幅改善 (zh_slow 9%→5%, en_slow 47%→42%), 耗时增加 ~18% (6.5s→7.7s)
- **Level 2** (heavy): CER 严重恶化 (39%→60%), 过度降噪破坏语音信号
- **结论**: Level 1 有轻微收益且开销可接受, 默认启用; Level 2 不可用

## Hotwords Effect (sherpa-large) – 2026-02-27

Using hotwords generated from Fcitx5 user dictionary (3077 words).
Sweep over boost score values (file score and `hotwords_score` set to same value):

```
                          zh_slow  zh_norm  en_slow  en_norm  lite_s  lite_n  heavy_s  heavy_n  AVG
no hotwords                  9%      7%      47%      93%      15%     40%      49%      54%    39%
score=0.1                    5%      7%      40%      96%      15%     40%      58%      58%    40%
score=0.5                    5%      7%      40%      96%      13%     40%      54%      58%    39%
score=1.0                    5%      7%      40%      96%      13%     40%      54%      60%    39%
score=1.5                    5%      7%      40%      96%      13%     40%      54%      64%    40%
score=2.0                    5%      9%      41%      96%      13%     40%      54%      64%    40%
```

- 开 hotwords 后一致改善: zh_slow (9→5%), en_slow (47→40%), lite_slow (15→13%)
- 一致退化: en_normal (93→96%), zh_en_heavy_normal 随 score 增大持续恶化 (54→58→64%)
- **score=0.5 最优**: AVG 持平 (39%), 拿到改善同时 heavy 退化最小
- score≥1.5 heavy 退化明显, score=2.0 连 zh_normal 也开始退化
