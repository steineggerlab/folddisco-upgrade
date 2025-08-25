#!/bin/zsh

STRUCT_DIR="./"
mkdir -p filtered_structures

all_files=(${(f)"$(find "$STRUCT_DIR" -type f)"})

while read id; do
    echo "Processing: $id" >> progress.log
    match=()
    for file in $all_files; do
        if [[ $(basename "$file") == "$id" ]]; then
            match="$file"
            break
        fi
    done

    if [[ -f "$match" ]]; then
        cp "$match" filtered_structures/
    else
        echo "Missing: $id" >> missing_ids.log
    fi
done < null_ids.txt


