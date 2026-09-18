import csv
import math
import os
import random
from collections import defaultdict

benchmark_type = "AFDB"
BASE_DIR = os.path.abspath(os.getcwd())

RESULT_DIR_ORG = "result_benchmark_v2"
RESULT_DIR_PREP = os.path.join(BASE_DIR, f"{RESULT_DIR_ORG}")
RESULT_DIR = os.path.join(RESULT_DIR_PREP, "folddisco_results_analyses")

TARGET_N = 250_000  # 최종적으로 남길 match 수
SEED = 42  # 재현성


def count_matching_residues(matching_residues_str: str) -> int:
    tokens = matching_residues_str.strip().split(",")
    valid = [t for t in tokens if t and t != "_"]
    return len(valid)


def safe_float(x: str) -> float:
    try:
        return float(x)
    except:
        return float("nan")


def preprocess_giant_file_stratified(
    input_path, output_path, target_n=TARGET_N, seed=SEED
):
    print(f"Filtering giant file with stratified sampling: {input_path} -> {output_path}")

    seen_pairs = {}

    with open(
        input_path, "r", encoding="utf-8", buffering=1024 * 1024 * 10
    ) as f_in:
        header_line = f_in.readline().strip()
        headers = (
            header_line.split("\t")
            if "\t" in header_line
            else header_line.split()
        )
        col = {name: i for i, name in enumerate(headers)}

        count = 0
        for line in f_in:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 1:
                parts = line.strip().split()

            try:
                q = parts[col["query_num"]]
                src = parts[col["source"]]
                tgt = parts[col["id"]]
                idf = safe_float(parts[col["idf_score"]])
                rmsd = safe_float(parts[col["rmsd"]])
                tm_score = safe_float(parts[col["tm_score"]])
                matching = parts[col["matching_residues"]]
                lab = int(parts[col["answer_label"]])
            except IndexError:
                continue

            L = count_matching_residues(matching)

            if (
                math.isnan(idf)
                or math.isnan(rmsd)
                or math.isnan(tm_score)
                or L <= 1
                or L > 32
            ):
                continue

            # (source, target) 중복 처리: 최고 L값 유지
            pair_key = (src, tgt)
            if pair_key not in seen_pairs or L > seen_pairs[pair_key][3]:
                seen_pairs[pair_key] = [
                    q,
                    src,
                    tgt,
                    L,
                    idf,
                    rmsd,
                    tm_score,
                    lab,
                ]

            count += 1
            if count % 10_000_000 == 0:
                print(f"Processed {count} lines...")

    # ==== L 값별 그룹화 (Stratification) ====
    print("\nGrouping unique pairs by L value...")
    rows_by_L = defaultdict(list)
    for row in seen_pairs.values():
        L_val = row[3]
        rows_by_L[L_val].append(row)

    del seen_pairs  # 메모리 회수

    total_unique = sum(len(group) for group in rows_by_L.values())
    print(f"Total Unique (source, target) pairs: {total_unique}")

    rng = random.Random(seed)

    # ==== L 분포 비율에 맞게 Stratified Downsampling ====
    sampled_rows = []

    if total_unique <= target_n:
        print(
            f"[WARN] Total unique rows ({total_unique}) <= target ({target_n}). Keeping all."
        )
        for L_val, group in rows_by_L.items():
            sampled_rows.extend(group)
    else:
        print(
            f"Performing Stratified Downsampling to exactly {target_n} rows..."
        )

        # 1차: 각 L 그룹별 프로포셔널 몫(Quota) 계산
        quotas = {}
        remainder_list = []

        for L_val, group in rows_by_L.items():
            # 비율 계산
            exact_quota = (len(group) / total_unique) * target_n
            base_quota = int(exact_quota)
            remainder = exact_quota - base_quota

            quotas[L_val] = base_quota
            remainder_list.append((remainder, L_val))

        # 2차: 버림(floor)으로 인해 남은 개수(Shortfall)를 소수점 잔여치가 높은 L 그룹에 1개씩 배분
        current_total = sum(quotas.values())
        shortfall = target_n - current_total

        # 소수점 잔여치 내림차순 정렬
        remainder_list.sort(key=lambda x: x[0], reverse=True)
        for i in range(shortfall):
            L_val = remainder_list[i][1]
            quotas[L_val] += 1

        # 3차: 각 L 그룹 내에서 무작위 셔플 후 지정된 Quota만큼 추출
        print("\n--- L Distribution Breakdown ---")
        print(f"{'L':>4} | {'Original Count':>14} | {'Sampled Count':>14} | {'Ratio':>8}")
        print("-" * 50)

        for L_val in sorted(rows_by_L.keys()):
            group = rows_by_L[L_val]
            quota = quotas[L_val]

            rng.shuffle(group)
            sampled_group = group[:quota]
            sampled_rows.extend(sampled_group)

            ratio = (len(sampled_group) / target_n) * 100
            print(
                f"{L_val:4d} | {len(group):14d} | {len(sampled_group):14d} | {ratio:7.2f}%"
            )

    # 전체 결과 셔플 (순서 편향 방지)
    rng.shuffle(sampled_rows)

    print("\nWriting stratified data to disk...")
    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        writer.writerow(
            [
                "query_num",
                "source",
                "target_id",
                "L",
                "idf",
                "rmsd",
                "tm_score",
                "label_raw",
            ]
        )
        writer.writerows(sampled_rows)

    n_pos = sum(1 for r in sampled_rows if r[7] == 1)
    print(
        f"\nFinal Done: {len(sampled_rows)} rows (label=1: {n_pos}, label=0: {len(sampled_rows) - n_pos})"
    )


# 실행
INPUT_PATH = f"{RESULT_DIR}/total_combined.txt"
FILTERED_PATH = f"{RESULT_DIR}/data_negative_AFDB_shuffle.txt"
preprocess_giant_file_stratified(INPUT_PATH, FILTERED_PATH)