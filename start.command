#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "缺少 .venv，请按 README.md 安装依赖。"
  read "?按回车关闭"
  exit 1
fi
exec .venv/bin/python app.py "$@"

