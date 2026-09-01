#!/usr/bin/env bash
set -euo pipefail

t="$1"
p="$2"
c="$3"
f="$4"

sed \
    -e "s/TTTTT/${t}/g" \
    -e "s/PPPPP/${p}/g" \
    -e "s/CCCCC/${c}/g" \
    EJE-inference.py > "$f"
