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
zh_slow.wav               32%    8.6s       2%   19.1s       9%    2.3s       9%    3.8s
zh_normal.wav             59%    9.8s       7%   21.0s      11%    2.3s       7%    3.7s
en_slow.wav              100%    7.0s     100%   21.1s      52%    2.2s      47%    3.6s
en_normal.wav            100%    2.5s     100%   16.0s     100%    2.0s      93%    3.2s
zh_en_lite_slow.wav       73%    8.6s      44%   21.4s      35%    2.2s      15%    3.6s
zh_en_lite_normal.wav     96%    4.0s      67%   18.4s      42%    2.1s      40%    3.2s
zh_en_heavy_slow.wav      85%    9.9s      75%   15.0s      60%    2.2s      49%    3.6s
zh_en_heavy_normal.wav    97%    5.1s      96%   13.2s      63%    2.1s      54%    3.4s
------------------------------------------------------------------------------------------
AVG CER (lower=better)    80%              61%              46%              39%
TOT time                       55.5s            145.2s            17.4s            28.1s
```

## Analysis

- **Overall**: sherpa-large (39%) > sherpa-small (46%) > vosk-large (61%) > vosk-small (80%)
- **Pure Chinese**: vosk-large 和 sherpa-large 并列最优 (2%/7% vs 9%/7%)
- **Pure English**: 均较差, sherpa-large 略好 (47%/93%), 首次在正常语速下有输出
- **Mixed zh-en**: sherpa-large 明显领先, 尤其 lite 慢速 (15% vs 35%), heavy 慢速 (49% vs 60%)
- **Speed**: sherpa-large (28s) 仍远快于 vosk-large (145s), 仅比 sherpa-small (17s) 慢 60%
