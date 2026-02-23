#!/bin/bash
# Record test audio files for speech recognition accuracy testing.
# Each prompt is read twice: normal speed then slow speed.
# Press Enter to start recording, Enter again to stop.

OUTDIR="$(dirname "$0")/test_wavs"
mkdir -p "$OUTDIR"

PROMPTS=(
    "zh|这个函数的作用是计算两个向量之间的余弦相似度，输入是两个浮点数列表，返回值是零到一之间的小数"
    "en|The function takes two parameters, a learning rate and a batch size, and returns the training loss as a floating point number"
    "zh_en_lite|我们用了一个 transformer 模型来做文本分类，训练了大概五十个 epoch 之后准确率达到了百分之九十五"
    "zh_en_heavy|我们的 backend 用的是 Python 写的，主要依赖 FastAPI 框架。数据库用的 PostgreSQL，通过 SQLAlchemy 做 ORM 映射"
)

SPEEDS=("normal" "slow")

for entry in "${PROMPTS[@]}"; do
    tag="${entry%%|*}"
    text="${entry#*|}"
    for speed in "${SPEEDS[@]}"; do
        filename="${tag}_${speed}.wav"
        filepath="$OUTDIR/$filename"
        echo ""
        echo "============================================"
        echo "  [$tag - $speed]"
        echo "  $text"
        echo "============================================"
        echo "  Press Enter to START recording..."
        read -r
        echo "  >>> Recording... Press Enter to STOP."
        pw-record --rate=16000 --channels=1 --format=s16 "$filepath" &
        REC_PID=$!
        read -r
        kill "$REC_PID" 2>/dev/null
        wait "$REC_PID" 2>/dev/null
        echo "  Saved: $filepath"
    done
done

echo ""
echo "All done! Recorded files:"
ls -lh "$OUTDIR"/*.wav
