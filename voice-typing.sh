#!/bin/bash
# Toggle nerd-dictation recording.
# First call: start daemon (loads model) and begin recording.
# Subsequent calls: toggle suspend/resume.
# Auto-suspends after 2s silence (model stays in memory).
export YDOTOOL_SOCKET="/run/user/$(id -u)/.ydotool_socket"
NERD_DICTATION="$HOME/Codes/VoiceTyping/nerd-dictation/nerd-dictation"
MODEL_DIR="$HOME/Codes/VoiceTyping/vosk-models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"

PID=$(pgrep -f "nerd-dictation begin" | head -1)

# Kill stale process whose audio child (parec) became zombie after system suspend
if [ -n "$PID" ] && ps --ppid "$PID" -o stat= 2>/dev/null | grep -q Z; then
    kill -9 "$PID" 2>/dev/null
    PID=""
fi

if [ -z "$PID" ]; then
    "$NERD_DICTATION" begin \
        --engine=sherpa \
        --vosk-model-dir="$MODEL_DIR" \
        --simulate-input-tool=YDOTOOL \
        --continuous \
        --timeout=2 &
else
    STATE=$(awk '/^State:/{print $2}' /proc/"$PID"/status 2>/dev/null)
    if [ "$STATE" = "T" ]; then
        kill -CONT "$PID"
    else
        kill -USR1 "$PID"
    fi
fi
