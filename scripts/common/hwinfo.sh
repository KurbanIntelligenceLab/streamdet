#!/bin/bash
# RF3-5: hardware/OS/library facts for the appendix (reproducibility checklist).
set -euo pipefail
export PYTHONPATH="Code:Code/streamdet:Code/vidaudit:${PYTHONPATH:-}"
{
  echo "key,value"
  echo "node,$(hostname)"
  echo "cpu_model,$(lscpu | grep 'Model name' | sed 's/Model name:\s*//' | sed 's/,/;/g')"
  echo "sockets,$(lscpu | awk '/^Socket\(s\)/{print $2}')"
  echo "cores_per_socket,$(lscpu | awk '/^Core\(s\) per socket/{print $4}')"
  echo "threads,$(nproc)"
  echo "os,$(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "kernel,$(uname -r)"
  python - <<'PY'
import platform, sys
print(f"python,{platform.python_version()}")
for m in ("torch","numpy","sklearn","pandas","av","transformers"):
    try:
        mod=__import__(m); print(f"{m},{mod.__version__}")
    except Exception as e:
        print(f"{m},ERR")
import av
print(f"ffmpeg_libavcodec,{av.library_versions.get('libavcodec')}")
PY
  echo "ffmpeg_cli,$(ffmpeg -version 2>/dev/null | head -1 | sed 's/,/;/g')"
} > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
cat "$OUT"
