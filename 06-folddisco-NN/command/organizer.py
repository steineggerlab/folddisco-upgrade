# 설정값 변경 (파일명을 본인 파일에 맞게 수정하세요)
input_file = "/fast2/jwyoon05/folddisco-dirs/folddisco-NN/result_benchmark_scope40/folddisco_results_analyses/total_combined.txt"  # 원본 파일명
output_file = "/fast2/jwyoon05/folddisco-dirs/folddisco-NN/result_benchmark_scope40/folddisco_results_analyses/total_combined_fixed.txt"  # 새로 저장할 파일명

# 변경할 새로운 헤더 (맨 끝에 줄바꿈 \n 포함)
new_header = (
    "query_num\tid\tsource\tnode_count\tidf_score\trmsd\ttm_score\t"
    "matching_residues\tanswer_label\n"
)

with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "w", encoding="utf-8"
) as f_out:

    # 1. 원본 파일의 첫 줄(기존 헤더)은 읽어서 지나치기
    f_in.readline()

    # 2. 새 파일에 새로운 헤더를 먼저 작성
    f_out.write(new_header)

    # 3. 남은 데이터는 한 줄씩 읽어서 바로바로 새 파일에 쓰기
    for line in f_in:
        f_out.write(line)

print(f"완료! 헤더가 정상적으로 교체되어 '{output_file}'로 저장되었습니다.")