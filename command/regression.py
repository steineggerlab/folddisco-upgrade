import numpy as np
import matplotlib.pyplot as plt

# 1. 설정: 길이 범위 (1 ~ 150)
l = np.linspace(1, 150, 300)

# 2. Mu 함수 (고정: 2차식 증가)
# l^2에 비례하여 커지는 값. Lambda가 이걸 버텨줘야 E-value가 정상 작동함.
mu = 0.01115566 * (l**2) + 0.06267775 * l + 18.8132141

# 3. 비교할 세 가지 Lambda 모델

# A. [기존] 지수 함수 (Exponential)
# - 문제: 너무 빨리 0이 됨
lam_exp = 0.67754035 * np.exp(-0.05775654 * l)

# B. [실패했던] 단순 유리 함수 (Simple Rational, 1/L)
# - 문제: 초반(Length 작을 때)에 값이 폭발함 -> 변별력 상실
# - 비교를 위해 Scale은 L=40 쯤에서 지수함수와 만나도록 조정
lam_rational = 10.0 / l 

# C. [제안] Shifted Power Law (Hybrid)
# - 전략: 초반엔 상수항 1 때문에 완만하고, 후반엔 1/L^c 로 버팀
# - 파라미터: c=1.5 (적당한 꼬리), a, b는 기존 지수함수와 시작점/기울기 일치시킴
a = 0.67754035
c = 1.5
b = 0.05775654 / c
lam_proposed = a / np.power(1 + b * l, c)

# 4. 시각화
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# --- 그래프 1: Lambda 값 자체의 변화 ---
ax[0].plot(l, lam_rational, 'r--', label='Rational (1/L)', alpha=0.5)
ax[0].plot(l, lam_exp, 'b-', label='Original (Exp)', linewidth=2)
ax[0].plot(l, lam_proposed, 'g-', label='Proposed (Shifted Power)', linewidth=3)
ax[0].set_ylim(0, 1.0) # 폭발하는 유리함수 잘라내고 보기 위해
ax[0].set_title("[Graph 1] Lambda Value vs Length")
ax[0].set_xlabel("Query Length (l)")
ax[0].set_ylabel("Lambda (Scale Factor)")
ax[0].legend()
ax[0].grid(True)
ax[0].text(5, 0.8, "Problem: Too High!\n(Loss of sensitivity)", color='red')
ax[0].text(100, 0.05, "Problem: Dies too fast!\n(Inversion happens)", color='blue')

# --- 그래프 2: Lambda * Mu (E-value의 핵심) ---
# 이 값이 "증가"해야 E-value가 정상적으로 작동함.
# 내려가면 -> "길이가 긴데 E-value가 나빠지는" 역전 현상 발생
prod_exp = lam_exp * mu
prod_rational = lam_rational * mu
prod_proposed = lam_proposed * mu

ax[1].plot(l, prod_rational, 'r--', label='Rational')
ax[1].plot(l, prod_exp, 'b-', label='Original (Exp)')
ax[1].plot(l, prod_proposed, 'g-', label='Proposed (Shifted Power)', linewidth=3)
ax[1].set_title("[Graph 2] Lambda * Mu (Stability Check)")
ax[1].set_xlabel("Query Length (l)")
ax[1].set_ylabel("Lambda * Mu value")
ax[1].legend()
ax[1].grid(True)

# 화살표로 설명
ax[1].annotate('CRASH! (Inversion)', xy=(40, prod_exp[80]), xytext=(60, 10),
             arrowprops=dict(facecolor='blue', shrink=0.05))
ax[1].annotate('Stable Growth', xy=(140, prod_proposed[-1]), xytext=(100, 25),
             arrowprops=dict(facecolor='green', shrink=0.05))

plt.tight_layout()
plt.show()
plt.savefig("results/lambda_models_comparison.png", dpi=300)
plt.close()