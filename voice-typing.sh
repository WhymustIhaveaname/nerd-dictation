#!/bin/bash
# Toggle nerd-dictation recording.
# First call: start daemon (loads model) and begin recording.
# Subsequent calls: toggle suspend/resume.
# Auto-suspends after 2s silence (model stays in memory).
export YDOTOOL_SOCKET="/run/user/$(id -u)/.ydotool_socket"
NERD_DICTATION="$HOME/Codes/VoiceTyping/nerd-dictation/nerd-dictation"
MODEL_DIR="$HOME/Codes/VoiceTyping/vosk-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

PID=$(pgrep -f "nerd-dictation begin" | head -1)

# Kill stale process whose audio child became zombie after system suspend
if [ -n "$PID" ] && ps --ppid "$PID" -o stat= 2>/dev/null | grep -q Z; then
    kill -9 "$PID" 2>/dev/null
    PID=""
    notify-send -t 3000 -u critical "Voice Typing" "Zombie process killed, restarting..."
fi

if [ -z "$PID" ]; then
    notify-send -t 3000 -u critical "Voice Typing" "Loading model..."
    "$NERD_DICTATION" begin \
        --engine=sherpa \
        --vosk-model-dir="$MODEL_DIR" \
        --simulate-input-tool=YDOTOOL \
        --continuous \
        --noise-reduction=1 \
        --timeout=3 &
else
    STATE=$(awk '/^State:/{print $2}' /proc/"$PID"/status 2>/dev/null)
    if [ "$STATE" = "T" ]; then
        kill -CONT "$PID"
        notify-send -t 1500 -u low "Voice Typing" "Recording"
    else
        kill -USR1 "$PID"
        notify-send -t 1500 -u low "Voice Typing" "Suspended"
    fi
fi
