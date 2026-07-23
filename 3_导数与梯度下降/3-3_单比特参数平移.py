import numpy as np
import matplotlib.pyplot as plt


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = [SX, SY, SZ]
PAULI_NAMES = ["σ_x", "σ_y", "σ_z"]
I = np.eye(2, dtype=complex)


def f_analytical(p1_idx, p2_idx, theta):
    P1 = PAULI[p1_idx]
    P2 = PAULI[p2_idx]

    ket0 = np.array([1.0 + 0j, 0.0 + 0j])
    psi = np.cos(theta / 2) * ket0 + 1j * np.sin(theta / 2) * (P1 @ ket0)
    expval = np.conjugate(psi).T @ (P2 @ psi)
    return np.real(expval)


def f_exact_prime(p1_idx, p2_idx, theta):
    P1 = PAULI[p1_idx]
    P2 = PAULI[p2_idx]


    commutator = 0.5j * (P2 @ P1 - P1 @ P2)
    ket0 = np.array([1.0 + 0j, 0.0 + 0j])
    psi = np.cos(theta / 2) * ket0 + 1j * np.sin(theta / 2) * (P1 @ ket0)
    return np.real(np.conjugate(psi).T @ (commutator @ psi))


def parameter_shift_deriv(p1_idx, p2_idx, theta):
    return (f_analytical(p1_idx, p2_idx, theta + np.pi / 2) -
            f_analytical(p1_idx, p2_idx, theta - np.pi / 2)) / 2.0


def numerical_deriv(p1_idx, p2_idx, theta, delta=1e-6):
    return (f_analytical(p1_idx, p2_idx, theta + delta) -
            f_analytical(p1_idx, p2_idx, theta - delta)) / (2 * delta)


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


        ax.plot(thetas / np.pi, f_vals, lw=2,
                label=r"$f(\theta)$", color="C0")

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
