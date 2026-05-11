"""
칼만 필터 NIS (Normalized Innovation Squared) 평가

05.02.02.Webcam-Kalman.py 실행 후 생성된 nis_log.csv를 분석합니다.

실행 방법:
    python 05.02.03.Kalman-NIS-Eval.py

평가 기준 (측정벡터 차원 n_z = 2):
    평균 NIS ≈ 2  → 정상 (R 파라미터 일관성 양호)
    평균 NIS >> 2 → R 과소 추정 (실제 센서 노이즈보다 R이 작음)
    평균 NIS << 2 → R 과대 추정 (실제 센서 노이즈보다 R이 큼)
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

CSV_PATH = "nis_log.csv"
NZ = 2      # 측정벡터 차원 (cx, cy)

# CSV 로드
frames, nis_values = [], []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frames.append(int(row["frame"]))
        nis_values.append(float(row["nis"]))

nis_values = np.array(nis_values)
mean_nis = np.mean(nis_values)

# χ² 95% 신뢰 구간 (자유도 NZ)
ci_low  = chi2.ppf(0.025, df=NZ)
ci_high = chi2.ppf(0.975, df=NZ)
print(ci_low, ci_low)

print(f"샘플 수   : {len(nis_values)}")
print(f"평균 NIS  : {mean_nis:.3f}  (기준 n_z = {NZ})")
print(f"95% CI   : [{ci_low:.2f}, {ci_high:.2f}]")
if mean_nis > ci_high:
    print("판정: R 과소 추정 → R 값을 키우세요")
elif mean_nis < ci_low:
    print("판정: R 과대 추정 → R 값을 줄이세요")
else:
    print("판정: 정상 (이노베이션 공분산 일관성 양호)")

# 시계열 + 히스토그램
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
fig.suptitle("Kalman Filter NIS Evaluation", fontsize=14)

# Time series
ax1.plot(frames, nis_values, color="steelblue", linewidth=0.8, alpha=0.7, label="NIS")
ax1.axhline(NZ,       color="green",  linestyle="--", linewidth=1.5, label=f"n_z = {NZ} (ideal)")
ax1.axhline(ci_low,   color="orange", linestyle=":",  linewidth=1.2, label=f"95% CI low  ({ci_low:.2f})")
ax1.axhline(ci_high,  color="orange", linestyle=":",  linewidth=1.2, label=f"95% CI high ({ci_high:.2f})")
ax1.axhline(mean_nis, color="red",    linestyle="-",  linewidth=1.5, label=f"Mean NIS ({mean_nis:.2f})")
ax1.set_xlabel("Frame")
ax1.set_ylabel("NIS")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Histogram
x = np.linspace(0, max(nis_values.max(), ci_high * 2), 300)
ax2.hist(nis_values, bins=40, density=True, color="steelblue", alpha=0.6, label="NIS distribution")
ax2.plot(x, chi2.pdf(x, df=NZ), color="red", linewidth=2, label=f"χ²(df={NZ}) theoretical")
ax2.axvline(mean_nis, color="red",    linestyle="--", linewidth=1.5, label=f"Mean ({mean_nis:.2f})")
ax2.axvline(ci_low,   color="orange", linestyle=":",  linewidth=1.2)
ax2.axvline(ci_high,  color="orange", linestyle=":",  linewidth=1.2, label=f"95% CI")
ax2.set_xlabel("NIS")
ax2.set_ylabel("Density")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("nis_eval.png", dpi=150)
plt.show()
print("그래프 저장: nis_eval.png")
