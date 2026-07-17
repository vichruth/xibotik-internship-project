#!/usr/bin/env bash
# Launch the assistant.
#
# The pipeline drives the onboard audio codec directly through ALSA (hw:1,0), which the
# software resampler in PortAudio can't share with PipeWire. So for the duration of the
# session we switch the analog card's profile off (releasing it from PipeWire) and restore
# it on exit. HDMI and other cards are untouched. Override detection with AUDIO_CARD=... .
# note: no `set -u` - conda's activation scripts are not nounset-safe.
source /home/vichruth/miniconda3/etc/profile.d/conda.sh
conda activate ml_vision
cd "$(dirname "$0")"

# ctranslate2 (faster-whisper's backend) links against CUDA 12's cuBLAS/cuDNN; this
# machine's system CUDA is 13 (via torch), so point the loader at the CUDA 12 libs
# pip installs into site-packages instead.
CU12_LIBS=$(python -c "import nvidia.cublas, nvidia.cudnn; print(list(nvidia.cublas.__path__)[0]+'/lib:'+list(nvidia.cudnn.__path__)[0]+'/lib')" 2>/dev/null)
[ -n "$CU12_LIBS" ] && export LD_LIBRARY_PATH="$CU12_LIBS:$LD_LIBRARY_PATH"
export HF_HUB_DISABLE_XET=1

CARD="${AUDIO_CARD:-}"
SAVED_PROFILE=""

restore_audio() {
    if [ -n "$CARD" ]; then
        local target="${SAVED_PROFILE:-output:analog-stereo+input:analog-stereo}"
        [ "$target" = "off" ] && target="output:analog-stereo+input:analog-stereo"
        pactl set-card-profile "$CARD" "$target" 2>/dev/null || true
    fi
}

if command -v pactl >/dev/null 2>&1; then
    [ -z "$CARD" ] && CARD=$(pactl list cards | awk '/^Card #/{n=""} /Name: alsa_card/{n=$2} /analog-stereo/{if(n){print n; exit}}')
    if [ -n "$CARD" ]; then
        SAVED_PROFILE=$(pactl list cards | awk -v c="$CARD" '/^Card #/{f=0} $0 ~ "Name: "c"$"{f=1} f&&/Active Profile:/{print $3; exit}')
        trap restore_audio EXIT INT TERM
        pactl set-card-profile "$CARD" off 2>/dev/null || true
        sleep 0.5
    fi
fi

python -u main.py
