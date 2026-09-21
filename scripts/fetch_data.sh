#!/usr/bin/env bash
# ECtHR judgments from LexGLUE, both tasks, all splits.
#
# Fetched in byte ranges because a plain GET over this link truncates silently:
# curl reports 200 and a plausible size, and the file has no parquet footer.
# The loop below reads Content-Length first and asks for exactly that many
# bytes, so a short file is a failure rather than a surprise later.
set -u
cd "$(dirname "$0")/../data" || exit 1

BASE="https://huggingface.co/api/datasets/coastalcph/lex_glue/parquet"
CHUNK=2000000

fetch() {
  local url="$1" out="$2" total i s e size
  total=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  if [ -z "$total" ]; then
    echo "  !! no content-length for $out"
    return 1
  fi
  : > "$out"
  i=0
  while :; do
    s=$((i * CHUNK)); e=$((s + CHUNK - 1))
    [ "$s" -ge "$total" ] && break
    curl -sL --max-time 180 -r "${s}-${e}" "$url" >> "$out" || return 1
    i=$((i + 1))
  done
  size=$(stat -c%s "$out")
  if [ "$size" -ne "$total" ]; then
    echo "  !! $out is $size bytes, expected $total"
    return 1
  fi
  echo "  ok $out  ${size} bytes"
}

for cfg in ecthr_a ecthr_b; do
  for split in train test validation; do
    echo "$cfg/$split"
    fetch "$BASE/$cfg/$split/0.parquet" "${cfg}_${split}.parquet"
  done
done

echo
echo "now run: python scripts/build_corpus.py"
