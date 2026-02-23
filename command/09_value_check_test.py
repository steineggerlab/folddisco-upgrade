# Python script to calculate E-values based on a scoring function and save the results to a TSV file. 
# The script includes tests for the monotonicity and upper bound of E-value calculations for a given scoring function.

import numpy as np
import unittest

# =============================================================================
# 계산 함수
# =============================================================================
def evalue_fitting_new(x, m, l):
    x_d = float(x)
    m_d = float(m)
    l_d = float(l)

    # 1. Mu, Lam 계산
    mu = 4.2161 * np.exp(l_d * 0.0489) + 3.6661
    lam = 0.2894 * np.exp(l_d * -0.0762) + 0.0316

    ref_db_size = 10546.0 
    k_val = np.exp(lam * mu) / ref_db_size
    real_search_space = m_d 

    e_val_raw = k_val * real_search_space * l_d * np.exp(-lam * x_d)
    e_val = (e_val_raw * real_search_space) / (e_val_raw + real_search_space)

    return e_val/(2), k_val

# =============================================================================
# Unit Test Class
# =============================================================================
class TestEValueCalculation(unittest.TestCase):
    # 전체 테스트 통과 여부를 추적하기 위한 클래스 변수
    test_results_status = []

    def setUp(self):
        self.m_size = 20000  
        self.scores = range(0, 31)
        self.lengths = range(2, 33)

    def _print_range(self, results, start, end, status):
        s_val = results[start]['val']
        e_val = results[end]['val']
        label = " [PASS] " if status == "PASS" else " [FAIL] "
        range_str = f"Score {int(s_val):2d} ~ {int(e_val):2d}"
        if start == end:
            range_str = f"Score {int(s_val):2d}       "
        print(f"{label} {range_str}", end="")
        if status == "FAIL":
            print(f" -> Info: {results[start].get('msg', '')}")
        else:
            print()

    def _generate_report(self, title, results):
        print("\n" + "="*60)
        print(f"[{title}]")
        print("="*60)
        failures = sum(1 for r in results if r['status'] == "FAIL")
        print(f"Total Checked: {len(results)}")
        print(f"Failed: {failures}\n")
        print("--- Range Report ---")

        if results:
            start_idx = 0
            current_status = results[0]['status']
            for i in range(1, len(results)):
                if results[i]['status'] != current_status:
                    self._print_range(results, start_idx, i-1, current_status)
                    start_idx = i
                    current_status = results[i]['status']
            self._print_range(results, start_idx, len(results)-1, current_status)
        
        print("="*60)
        return failures

    def test_e_value_monotonicity_report(self):
        score_results = []
        for x in self.scores:
            is_failed = False
            prev_e = -1.0
            for l in self.lengths:
                curr_e, _ = evalue_fitting_new(x, self.m_size, l)
                if prev_e != -1.0 and curr_e <= prev_e:
                    is_failed = True
                    break
                prev_e = curr_e
            score_results.append({'val': x, 'status': "FAIL" if is_failed else "PASS"})

        fail_count = self._generate_report("Test 1 Analysis: Monotonicity (Length vs E-value)", score_results)
        self.test_results_status.append(fail_count == 0)
        if fail_count > 0:
            self.fail(f"Monotonicity violation found.")

    def test_e_value_upper_bound_report(self):
        score_results = []
        limit = float(self.m_size)
        for x in self.scores:
            is_failed = False
            max_e_found = 0.0
            for l in self.lengths:
                e_val, _ = evalue_fitting_new(x, limit, l)
                if e_val > limit + 1e-7:
                    is_failed = True
                    max_e_found = e_val
                    break
            status = "PASS"
            msg = ""
            if is_failed:
                status = "FAIL"
                msg = f"E-value({max_e_found:.2f}) exceeded limit({limit})"
            score_results.append({'val': x, 'status': status, 'msg': msg})

        fail_count = self._generate_report("Test 2 Analysis: Upper Bound (E-value <= DB Size)", score_results)
        self.test_results_status.append(fail_count == 0)
        if fail_count > 0:
            self.fail(f"Upper bound violation found.")

if __name__ == "__main__":
    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=0)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEValueCalculation)
    result = runner.run(suite)

    # 최종 결과 출력 섹션
    print("\n\n" + "#"*60)
    if result.wasSuccessful():
        print("#" + " "*58 + "#")
        print("#" + "  ALL TESTS PASSED SUCCESSFULLY! ".center(58) + "#")
        print("#" + " "*58 + "#")
    else:
        print("#" + " "*58 + "#")
        print("#" + " SOME TESTS FAILED. CHECK THE REPORT ABOVE. ".center(58) + "#")
        print("#" + " "*58 + "#")
    print("#"*60 + "\n")