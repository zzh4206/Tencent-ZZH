import numpy as np
import matplotlib.pyplot as plt


sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

PAULI = {"σx": sigma_x, "σy": sigma_y, "σz": sigma_z}


def matrix_exp_direct(P, theta):
    from scipy.linalg import expm
    return expm(1j * theta * P)


def matrix_exp_formula(P, theta):
    return np.cos(theta) * I + 1j * np.sin(theta) * P


def matrix_exp_taylor(P, theta, n_terms=20):
    result = np.zeros((2, 2), dtype=complex)
    term = np.eye(2, dtype=complex)
    for k in range(n_terms):
        result += term
        term = term @ (1j * theta * P) / (k + 1)
    return result


theta_test = np.pi / 3
print("=" * 60)
print("数值验证：e^{iθP} = cos(θ)I + i sin(θ)P")
print("=" * 60)

for name, P in PAULI.items():
    direct = matrix_exp_direct(P, theta_test)
    formula = matrix_exp_formula(P, theta_test)
    taylor = matrix_exp_taylor(P, theta_test)
    err_direct = np.max(np.abs(direct - formula))
    err_taylor = np.max(np.abs(taylor - formula))
    print(f"\n{name}:")
    print(f"  expm vs 公式 最大误差: {err_direct:.2e}")
    print(f"  泰勒(20项) vs 公式 最大误差: {err_taylor:.2e}")


print("\n" + "=" * 60)
print("验证 P² = I：")
for name, P in PAULI.items():
    print(f"  {name}² = I ?  max|P² - I| = {np.max(np.abs(P @ P - I)):.2e}")


thetas = np.linspace(0, 2 * np.pi, 100)
fig, axes = plt.subplots(3, 3, figsize=(12, 10))

for row, (name, P) in enumerate(PAULI.items()):
    diffs = []
    for theta in thetas:
        direct = matrix_exp_direct(P, theta)
        formula = matrix_exp_formula(P, theta)
        diffs.append(np.max(np.abs(direct - formula)))
    for col in range(3):
        ax = axes[row, col]
        theta_fixed = np.pi * (col + 1) / 2
        M = matrix_exp_formula(P, theta_fixed)
        ax.imshow(np.abs(M), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="red" if np.abs(M[i, j]) > 0.5 else "black")
        ax.set_title(f"{name}, θ={theta_fixed/np.pi:.1f}π")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

plt.suptitle(r"$e^{i\theta P}$ 矩阵元素 (绝对值)", fontsize=14)
plt.tight_layout()
plt.savefig("2-2_矩阵指数.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("结论：")
print("1. e^{iθP} = cos(θ)I + i sin(θ)P 对任意泡利矩阵精确成立。")
print("2. 成立条件：P² = I（对合矩阵）。")
print("3. 更一般地，对任意满足 P² = I 的方阵，该欧拉公式均成立。")
