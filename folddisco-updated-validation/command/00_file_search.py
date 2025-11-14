import os

ROOT = "result/folddisco_results_raw"
TARGET = "output_1twfB04.txt"

for root, dirs, files in os.walk(ROOT):
    if TARGET in files:
        print("FOUND:", os.path.join(root, TARGET))