import numpy as np
from functools import reduce


def kron(*matrices):
    return reduce(np.kron, matrices)


I_mat = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def build_H_full_matrix(n):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)


    for i in range(n):
        ops = [I_mat] * n
        ops[i] = Z
        H += kron(*ops)


    for i in range(n - 1):
        ops = [I_mat] * n
        ops[i] = X
        ops[i + 1] = X
        H += kron(*ops)

    return H


def compute_expectation(n):

    analytic = n


    dim = 2 ** n
    v0 = np.zeros(dim, dtype=complex)
    v0[0] = 1.0

    if n <= 6:
        H = build_H_full_matrix(n)
        numerical = np.real(np.conjugate(v0).T @ (H @ v0))
        err = abs(numerical - analytic)
        print(f"  n={n}: 解析={analytic}, 数值={numerical:.6f}, 误差={err:.2e}")
    else:
        print(f"  n={n}: 解析={analytic}（矩阵维度 {dim}×{dim} 太大，仅给解析结果）")

    return analytic


print("=" * 60)
print("⟨0...0| H |0...0⟩ 的期望值")
print("=" * 60)
for n_test in [2, 3, 4, 5, 6, 10]:
    compute_expectation(n_test)


print("\n" + "=" * 60)
print("n=2 时的 H 矩阵（4×4）：")
print("=" * 60)
H2 = build_H_full_matrix(2)
print(np.real(H2))

print("\n说明：H 是对角线 + 非对角线的结构。")
print("对角元来自 Z_i 项，非对角元来自 X_i X_{i+1}（自旋翻转）项。")


import matplotlib.pyplot as plt

H3 = build_H_full_matrix(3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
im1 = ax1.imshow(np.real(H3), cmap="RdBu_r", interpolation="nearest")
ax1.set_title("Re[H] (n=3)")
plt.colorbar(im1, ax=ax1)
im2 = ax2.imshow(np.imag(H3), cmap="RdBu_r", interpolation="nearest")
ax2.set_title("Im[H] (n=3)")
plt.colorbar(im2, ax=ax2)
plt.suptitle(r"$H = \sum_i Z_i + \sum_i X_i X_{i+1}$ (n=3)")
plt.tight_layout()
plt.savefig("2-4_张量积.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n结论：⟨0...0|H|0...0⟩ = n，即等于量子比特数。")
print("Z 项贡献 n，XX 耦合项在 |0...0⟩ 下期望为 0。")
