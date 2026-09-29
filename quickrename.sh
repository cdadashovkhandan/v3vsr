#!/usr/bin/env bash
set -euo pipefail

root="${1:-.}"

find "$root" -mindepth 2 -type f -name 'image-*.png' -print0 |
while IFS= read -r -d '' file; do
    dir=$(dirname -- "$file")
    name=$(basename -- "$file")

    # Extract the number from image-N.png
    if [[ "$name" =~ ^image-([0-9]+)\.png$ ]]; then
        number=$((10#${BASH_REMATCH[1]} - 1))
        target="$dir/$number.png"

        if [[ -e "$target" ]]; then
            printf 'Skipping: target already exists: %s\n' "$target" >&2
            continue
        fi

        mv -- "$file" "$target"
        printf '%s -> %s\n' "$file" "$target"
    fi
done
