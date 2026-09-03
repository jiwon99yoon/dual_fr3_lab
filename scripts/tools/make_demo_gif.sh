#!/usr/bin/env bash
# 데모 영상(webm/mp4) -> README용 GIF 변환.
#
# 프로젝트 데모 gif는 전부 Isaac Sim 뷰포트 화면 녹화(상단 툴바·FPS 오버레이 포함)를
# 그대로 gif로 만든 것이다 (crop 없음). 기존 demos/*.gif 규격에 맞춘 기본값:
#   폭 640px, 15fps, palette 최적화(diff 통계 + bayer dither) — 화질/용량 균형.
#
# 사용법:
#   scripts/tools/make_demo_gif.sh <입력.webm|mp4> [출력.gif] [폭] [fps]
#
# 예:
#   scripts/tools/make_demo_gif.sh ~/Videos/스크린캐스트/Isaac-Reach-DualFR3-Play-v0.webm
#     -> demos/dual_fr3_reach.gif ... 가 아니라 입력 파일명 기반 demos/<stem>.gif 로 저장
#   scripts/tools/make_demo_gif.sh in.webm demos/dual_fr3_reach.gif 640 15
#
# 용량이 크면 폭 600 / fps 12 로 낮춘다. 커밋 시 *.gif 는 git-lfs 추적(README Notes 참조).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <input.webm|mp4> [output.gif] [width=640] [fps=15]" >&2
    exit 1
fi

SRC="$1"
if [[ ! -f "$SRC" ]]; then
    echo "error: 입력 파일 없음: $SRC" >&2
    exit 1
fi

# 출력 경로 기본값: repo 루트의 demos/<입력파일 stem>.gif
STEM="$(basename "${SRC%.*}")"
OUT="${2:-demos/${STEM}.gif}"
WIDTH="${3:-640}"
FPS="${4:-15}"

command -v ffmpeg >/dev/null || { echo "error: ffmpeg 미설치" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"

PAL="$(mktemp --suffix=.png)"
trap 'rm -f "$PAL"' EXIT

echo "[1/2] palette 생성 (width=$WIDTH fps=$FPS)"
ffmpeg -v error -i "$SRC" \
    -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" \
    -y "$PAL"

echo "[2/2] gif 인코딩 -> $OUT"
ffmpeg -v error -i "$SRC" -i "$PAL" \
    -lavfi "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3" \
    -y "$OUT"

SIZE="$(du -h "$OUT" | cut -f1)"
DIM="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$OUT" 2>/dev/null)"
echo "done: $OUT  (${DIM}, ${SIZE})"
