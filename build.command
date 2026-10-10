#!/bin/zsh
set -eu
cd "$(dirname "$0")"
export PYINSTALLER_CONFIG_DIR="$PWD/build/pyinstaller-cache"
.venv/bin/pyinstaller --noconfirm --windowed --name SmallARC \
  --target-architecture arm64 --paths src app.py
print "已生成 $PWD/dist/SmallARC.app"

