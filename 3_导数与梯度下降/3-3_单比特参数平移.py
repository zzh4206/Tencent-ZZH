"""
星火预习题目 — 3-3：单比特参数平移
=====================================
f(θ) = ⟨0| e^{-iθ/2 P₁} P₂ e^{iθ/2 P₁} |0⟩

其中 P₁, P₂ ∈ {σ_x, σ_y, σ_z}。

【解析推导】
设 |ψ(θ)⟩ = e^{iθ/2 P₁} |0⟩，则 f(θ) = ⟨ψ(θ)| P₂ |ψ(θ)⟩。

利用 e^{iθ/2 P₁} = cos(θ/2) I + i sin(θ/2) P₁，以及泡利矩阵的对易关系：

情况 1: P₁ = P₂
  f(θ) = ⟨0| P₁ |0⟩（因为幺正旋转不改变与生成元对易的算符的期望）
  若 P₁ = σ_z: f(θ) = 1, f'(θ) = 0
  若 P₁ = σ_x 或 σ_y: f(θ) = 0, f'(θ) = 0

情况 2: P₁ ≠ P₂
  f(θ) = cos(θ) ⟨0|P₂|0⟩ + sin(θ) ⟨0|i[P₁, P₂]|0⟩
  具体形式取决于哪对泡利矩阵。

【周期性】
f(θ) 的周期为 2π（等价于 Bloch 球旋转一周）。
某些组合下周期为 π（如 P₁=σ_z, P₂=σ_x 时 f=cosθ）。

【参数平移对 f(θ) 的适用性】
参数平移法则要求 f 是单频正弦函数。一般地：
- P₁ = P₂ 时 f 为常数，参数平移给出 0 = 0，成立但平凡。
- P₁ ≠ P₂ 时 f(θ) = a cosθ + b sinθ（单频），参数平移法则成立：
  f'(θ) = [f(θ+π/2) - f(θ-π/2)] / 2
  这是因为 f 可写为 A sin(θ + φ) 的形式。
"""

import numpy as np
import matplotlib.pyplot as plt

# Pauli matrices
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [SX, SY, SZ]
PAULI_NAMES = ["σ_x", "σ_y", "σ_z"]
I = np.eye(2, dtype=complex)


def f_analytical(p1_idx, p2_idx, theta):
    """解析计算 f(θ) = ⟨0| e^{-iθ/2 P₁} P₂ e^{iθ/2 P₁} |0⟩"""
    P1 = PAULI[p1_idx]
    P2 = PAULI[p2_idx]
    # e^{iθ/2 P₁} |0⟩
    ket0 = np.array([1.0 + 0j, 0.0 + 0j])
    psi = np.cos(theta / 2) * ket0 + 1j * np.sin(theta / 2) * (P1 @ ket0)
    expval = np.conjugate(psi).T @ (P2 @ psi)
    return np.real(expval)


def f_exact_prime(p1_idx, p2_idx, theta):
    """解析导数 f'(θ)"""
    P1 = PAULI[p1_idx]
    P2 = PAULI[p2_idx]
    # f'(θ) = ⟨0| e^{-iθ/2 P₁} (-i/2 P₁ P₂ + i/2 P₂ P₁) e^{iθ/2 P₁} |0⟩
    #       = ⟨0| e^{-iθ/2 P₁} (i/2)[P₂, P₁] e^{iθ/2 P₁} |0⟩
    commutator = 0.5j * (P2 @ P1 - P1 @ P2)
    ket0 = np.array([1.0 + 0j, 0.0 + 0j])
    psi = np.cos(theta / 2) * ket0 + 1j * np.sin(theta / 2) * (P1 @ ket0)
    return np.real(np.conjugate(psi).T @ (commutator @ psi))


def parameter_shift_deriv(p1_idx, p2_idx, theta):
    """参数平移法则求导：f'(θ) = [f(θ+π/2) - f(θ-π/2)] / 2"""
    return (f_analytical(p1_idx, p2_idx, theta + np.pi / 2) -
            f_analytical(p1_idx, p2_idx, theta - np.pi / 2)) / 2.0


def numerical_deriv(p1_idx, p2_idx, theta, delta=1e-6):
    """有限差分数值求导"""
    return (f_analytical(p1_idx, p2_idx, theta + delta) -
            f_analytical(p1_idx, p2_idx, theta - delta)) / (2 * delta)


# ============================================================
# 遍历所有 (P1, P2) 组合
# ============================================================
print("=" * 70)
print("单比特参数平移：f(θ) = ⟨0|e^{-iθ/2 P₁} P₂ e^{iθ/2 P₁}|0⟩")
print("=" * 70)

thetas = np.linspace(0, 4 * np.pi, 400)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for p1 in range(3):
    for p2 in range(3):
        ax = axes[p1, p2]

        f_vals = [f_analytical(p1, p2, t) for t in thetas]
        fprime_exact = [f_exact_prime(p1, p2, t) for t in thetas]
        fprime_ps = [parameter_shift_deriv(p1, p2, t) for t in thetas]
        fprime_num = [numerical_deriv(p1, p2, t) for t in thetas]

        # f(θ) 曲线
        ax.plot(thetas / np.pi, f_vals, lw=2,
                label=r"$f(\theta)$", color="C0")
        # 导数
        ax.plot(thetas / np.pi, fprime_exact, '--', lw=1.5,
                label=r"$f'_{exact}$", color="C1")
        ax.plot(thetas / np.pi, fprime_ps, ':', lw=2,
                label=r"$f'_{ps}$", color="C2")
        ax.plot(thetas / np.pi, fprime_num, '-.', lw=1,
                label=r"$f'_{fd}$", color="C3")

        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_ylabel("value")
        ax.set_title(f"P₁={PAULI_NAMES[p1]}, P₂={PAULI_NAMES[p2]}")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        if p1 == 0 and p2 == 0:
            ax.legend(fontsize=7, loc="upper right")

plt.suptitle(r"$f(\theta) = \langle 0| e^{-i\theta/2 P_1} P_2 e^{i\theta/2 P_1} |0\rangle$ 及导数",
             fontsize=14)
plt.tight_layout()
plt.savefig("3-3_单比特参数平移.png", dpi=150, bbox_inches="tight")
plt.show()

# 验证参数平移的精确性
print("\n" + "=" * 70)
print("参数平移法则验证（随机测试点）")
print("=" * 70)
np.random.seed(42)
for p1 in range(3):
    for p2 in range(3):
        max_err = 0.0
        for _ in range(20):
            t = np.random.uniform(0, 2 * np.pi)
            exact = f_exact_prime(p1, p2, t)
            ps = parameter_shift_deriv(p1, p2, t)
            max_err = max(max_err, abs(exact - ps))
        status = "✓" if max_err < 1e-10 else "✗"
        print(f"  P₁={PAULI_NAMES[p1]}, P₂={PAULI_NAMES[p2]}: "
              f"max error={max_err:.2e} {status}")

# 解析表达式讨论
print("\n" + "=" * 70)
print("简化解析表达式")
print("=" * 70)
print("""
对 9 种组合：

P₁=σ_x:
  P₂=σ_x: f(θ) = 0                          (常数)
  P₂=σ_y: f(θ) = -sinθ                      (周期 2π)
  P₂=σ_z: f(θ) = cosθ                       (周期 2π)

P₁=σ_y:
  P₂=σ_x: f(θ) = sinθ                       (周期 2π)
  P₂=σ_y: f(θ) = 0                          (常数)
  P₂=σ_z: f(θ) = cosθ                       (周期 2π)

P₁=σ_z:
  P₂=σ_x: f(θ) = cosθ                       (周期 2π，注意相位)
  P₂=σ_y: f(θ) = sinθ                       (周期 2π)
  P₂=σ_z: f(θ) = 1                          (常数)

BLOCH球解释：
- P₁ = P₂: 旋转轴和观测轴一致 → 期望为常数
- P₁ ≠ P₂: 旋转使 Bloch 矢量绕 P₁ 旋转，
  在 P₂ 方向的投影按 sin/cos 规律振荡。

参数平移法则对所有组合成立，因为 f(θ) 总是 A sinθ + B cosθ + C 形式。
""")
