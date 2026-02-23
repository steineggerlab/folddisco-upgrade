#Python script to search for a specific file within a directory and its subdirectories, printing the full path if found. (Just for reference)

import os

ROOT = "result/folddisco_results_raw"
TARGET = "output_1twfB04.txt"

for root, dirs, files in os.walk(ROOT):
    if TARGET in files:
        print("FOUND:", os.path.join(root, TARGET))