#!/usr/bin/env bash
# 『自給自足仙人 e:鴨長明』ワンコマンド起動
# 監督モード: vLLMの仙人(羅漢級)を承認制(毎日停止→[次の日へ])で観戦・指揮する
cd "$(dirname "$0")"
exec python3 -m spl pixel --llm --cassette "Qwen仙人vLLM" --speed 1 --book "$@"
