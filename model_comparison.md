# Model Comparison – 2026-02-23

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
                          vosk-small        vosk-large       sherpa-small
wav                       CER    time      CER    time      CER    time
------------------------------------------------------------------------
zh_slow.wav               32%    5.9s       2%   13.3s       9%    2.6s
zh_normal.wav             59%    6.6s       7%   15.2s      11%    2.4s
en_slow.wav              100%    4.7s     100%   15.7s      52%    2.4s
en_normal.wav            100%    1.7s     100%   11.9s     100%    2.3s
zh_en_lite_slow.wav       73%    6.7s      44%   15.4s      35%    2.4s
zh_en_lite_normal.wav     96%    2.7s      67%   13.6s      42%    2.3s
zh_en_heavy_slow.wav      85%    6.5s      75%   15.0s      60%    3.2s
zh_en_heavy_normal.wav    97%    3.3s      96%   15.5s      63%    3.1s
------------------------------------------------------------------------
AVG CER (lower=better)    80%              61%              46%
TOT time                       38.1s            115.6s            20.7s
```

## Analysis

- **Overall**: sherpa-small (46%) > vosk-large (61%) > vosk-small (80%)
- **Pure Chinese**: vosk-large 最优 (2%/7%), sherpa-small 接近 (9%/11%)
- **Pure English**: 均较差, sherpa-small 慢速勉强可用 (52%)
- **Mixed zh-en**: 差距最大, sherpa-small 远优于 vosk, 尤其正常语速
- **Speed**: sherpa-small 5-6x faster than vosk-large

