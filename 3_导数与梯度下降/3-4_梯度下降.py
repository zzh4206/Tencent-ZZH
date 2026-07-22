"""
星火预习题目 — 3-4：梯度下降
===============================
利用 TensorFlow 后端，通过梯度下降寻找 3-3 中 f(θ) 的最小值。

梯度下降算法：
  θ_{k+1} = θ_k - η · f'(θ_k)

其中 η 为学习率。

注意：3-3 中的 f 是单变量函数，用梯度下降对其进行优化是
最基本的机器学习练习。同时也可以通过解析方式验证结果。
"""

import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


# ============================================================
# 用 TC 线路定义 f(θ)
# ============================================================
def make_f(p1_idx, p2_idx):
    """
    构造 f(θ) = ⟨0| e^{-iθ/2 P₁} P₂ e^{iθ/2 P₁} |0⟩
    的 TC 线路版本。
    p1_idx, p2_idx: 0→x, 1→y, 2→z
    """
    gate_map = {0: "rx", 1: "ry", 2: "rz"}
    obs_map = {0: "x", 1: "y", 2: "z"}

    def f(theta):
        c = tc.Circuit(1)
        gate = getattr(c, gate_map[p1_idx])
        gate(0, theta=theta)
        obs = obs_map[p2_idx]
        return K.real(c.expectation_ps(**{obs: [0]}))

    return f


def gradient_descent(f, theta0, learning_rate=0.1, n_steps=50, tol=1e-8):
    """
    用数值微分梯度做梯度下降。
    返回 (theta_history, f_history)
    """
    theta = K.convert_to_tensor(theta0, dtype=K.float32)
    grad_f = K.grad(f)

    theta_history = [theta.numpy()]
    f_history = [f(theta).numpy()]

    for step in range(n_steps):
        g = grad_f(theta)
        theta_new = theta - learning_rate * g

        theta = theta_new
        theta_history.append(theta.numpy())
        f_history.append(f(theta).numpy())

        # 检查收敛
        if abs(g.numpy()) < tol:
            print(f"  收敛于 step {step + 1}，梯度 = {g.numpy():.2e}")
            break

    return np.array(theta_history), np.array(f_history)


# ============================================================
# 对所有 9 种 (P₁, P₂) 组合做梯度下降
# ============================================================
PAULI_NAMES = ["σ_x", "σ_y", "σ_z"]
print("=" * 70)
print("梯度下降寻找 f(θ) 最小值")
print("=" * 70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for p1 in range(3):
    for p2 in range(3):
        f = make_f(p1, p2)

        # 多个初始点尝试
        best_min = float("inf")
        best_theta = None
        best_history = None

        for theta0 in [0.5, 1.5, 2.5, 4.0, 5.0]:
            th_hist, f_hist = gradient_descent(f, theta0,
                                                learning_rate=0.1,
                                                n_steps=100)
            if f_hist[-1] < best_min:
                best_min = f_hist[-1]
                best_theta = th_hist[-1]
                best_history = (th_hist, f_hist)

        print(f"P₁={PAULI_NAMES[p1]}, P₂={PAULI_NAMES[p2]}: "
              f"min f = {best_min:.6f} at θ ≈ {best_theta:.4f}")

        # 绘图
        ax = axes[p1, p2]
        thetas_plot = np.linspace(0, 4 * np.pi, 500)
        f_vals_plot = [f(float(t)).numpy() for t in thetas_plot]
        ax.plot(thetas_plot / np.pi, f_vals_plot, lw=1, alpha=0.5, label=r"$f(\theta)$")

        th_h, f_h = best_history
        ax.plot(th_h / np.pi, f_h, 'o-', markersize=3, lw=1, color="red",
                label="GD path")
        ax.plot(th_h[-1] / np.pi, f_h[-1], '*', markersize=12, color="red",
                markeredgecolor="black", label=f"min={f_h[-1]:.3f}")

        ax.set_xlabel(r"$\theta/\pi$")
        ax.set_ylabel(r"$f(\theta)$")
        ax.set_title(f"P₁={PAULI_NAMES[p1]}, P₂={PAULI_NAMES[p2]}")
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)

plt.suptitle("梯度下降寻找 f(θ) 最小值", fontsize=14)
plt.tight_layout()
plt.savefig("3-4_梯度下降.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 学习率对收敛的影响
# ============================================================
print("\n" + "=" * 70)
print("学习率影响（以 P₁=σ_x, P₂=σ_z, f=cosθ 为例）")
print("=" * 70)

f_cos = make_f(0, 2)  # f(θ) = cosθ

fig2, (ax_lr, ax_conv) = plt.subplots(1, 2, figsize=(12, 4))

for lr in [0.01, 0.1, 0.5, 1.0, 1.5]:
    th_h, f_h = gradient_descent(f_cos, 1.0, learning_rate=lr, n_steps=50)
    label = f"η={lr:.2f}, final f={f_h[-1]:.3f}"
    ax_conv.plot(range(len(f_h)), f_h, '.-', markersize=2, lw=1, label=label)

ax_conv.set_xlabel("Step")
ax_conv.set_ylabel(r"$f(\theta)$")
ax_conv.set_title("Convergence for different learning rates")
ax_conv.legend(fontsize=8)
ax_conv.grid(True, alpha=0.3)
ax_conv.axhline(y=-1.0, color='gray', linestyle='--', alpha=0.5, label="Global min")

# 最终 θ 与全局最小值
ax_lr.axis("off")
text = ("Global minima:\n"
        "P₁=σ_x: P₂=σ_z: f(θ)=cosθ → min -1 at θ=π,3π,...\n"
        "P₁=σ_y: P₂=σ_z: f(θ)=cosθ → min -1 at θ=π,3π,...\n"
        "P₁=σ_x: P₂=σ_y: f(θ)=-sinθ → min -1 at θ=π/2,5π/2,...\n"
        "P₁=σ_y: P₂=σ_x: f(θ)=sinθ → min -1 at θ=3π/2,7π/2,...\n"
        "P₁=σ_z: P₂=σ_x: f(θ)=cosθ → min -1 at θ=π,3π,...\n"
        "P₁=σ_z: P₂=σ_y: f(θ)=sinθ → min -1 at θ=3π/2,7π/2,...")
ax_lr.text(0, 1, text, transform=ax_lr.transAxes, fontsize=10,
           fontfamily="monospace", verticalalignment="top")

plt.tight_layout()
plt.savefig("3-4_学习率影响.png", dpi=150, bbox_inches="tight")
plt.show()
