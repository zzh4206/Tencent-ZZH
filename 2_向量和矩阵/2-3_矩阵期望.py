import numpy as np
import matplotlib.pyplot as plt


SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LIST = [SX, SY, SZ]
PAULI_NAMES = ["σ_x", "σ_y", "σ_z"]
I = np.eye(2, dtype=complex)


def expectation(P_idx, Q_idx, theta):
    P = PAULI_LIST[P_idx]
    Q = PAULI_LIST[Q_idx]


    v0 = np.array([1.0 + 0j, 0.0 + 0j])

    v = np.cos(theta / 2) * v0 + 1j * np.sin(theta / 2) * (P @ v0)


    expval = np.conjugate(v).T @ (Q @ v)
    return np.real(expval)


thetas = np.linspace(0, 4 * np.pi, 400)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))

for p_idx in range(3):
    for q_idx in range(3):
        ax = axes[p_idx, q_idx]
        vals = [expectation(p_idx, q_idx, t) for t in thetas]
        ax.plot(thetas / np.pi, vals)
        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_ylabel(r"$\langle Q \rangle$")
        ax.set_title(f"P={PAULI_NAMES[p_idx]}, Q={PAULI_NAMES[q_idx]}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

plt.suptitle(r"$\langle Q \rangle = \langle 0| e^{-i\theta/2 P} Q e^{i\theta/2 P} |0\rangle$", fontsize=14)
plt.tight_layout()
plt.savefig("2-3_矩阵期望.png", dpi=150, bbox_inches="tight")
plt.show()


print("=" * 60)
print("规律讨论：")
print("=" * 60)
print("""
1. P = σ_z:
   e^{iθ/2 σ_z}|0⟩ = e^{iθ/2}|0⟩，仅全局相位，不改变物理态。
   → ⟨σ_z⟩ 恒为 1（始终在 |0⟩）
   → ⟨σ_x⟩ = ⟨σ_y⟩ = 0（|0⟩ 是 σ_z 本征态）

2. P = σ_x:
   e^{iθ/2 σ_x}|0⟩ 在 Bloch 球上绕 x 轴旋转。
   → ⟨σ_x⟩ 恒为 0（x 轴旋转不动 x 分量）
   → ⟨σ_y⟩, ⟨σ_z⟩ 以周期 2π 振荡

3. P = σ_y:
   e^{iθ/2 σ_y}|0⟩ 在 Bloch 球上绕 y 轴旋转。
   → ⟨σ_y⟩ 恒为 0（y 轴旋转不动 y 分量）
   → ⟨σ_x⟩, ⟨σ_z⟩ 以周期 2π 振荡

4. P = Q 时: 期望值恒为 0（绕自身轴旋转时，初始态是垂直的）

5. 所有期望值关于 θ 是周期的，周期为 2π（矢量旋转一周）。
""")
