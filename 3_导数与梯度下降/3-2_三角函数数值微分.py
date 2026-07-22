"""
星火预习题目 — 3-2：三角函数数值微分（参数平移法则）
======================================================
f(x) = A sin(x + B) + C

【解析推导】
精确导数：f'(x) = A cos(x + B)

中心差分：(f(x+δ) - f(x-δ)) / τ

要使得数值差分严格等于解析导数，需：
  [A sin(x+δ+B) + C - A sin(x-δ+B) - C] / τ = A cos(x+B)
  → A [sin(x+δ+B) - sin(x-δ+B)] / τ = A cos(x+B)
  → 2A sin(δ) cos(x+B) / τ = A cos(x+B)
  → 2 sin(δ) / τ = 1
  → τ = 2 sin(δ)

当取 δ = π/2 时：τ = 2 sin(π/2) = 2
此时：f'(x) = [f(x+π/2) - f(x-π/2)] / 2 = A cos(x+B)  ✓

这就是"参数平移法则"（parameter shift rule）：
对于单频正弦函数 f(x) = A sin(x+B) + C，
取 δ = π/2, τ = 2，差分公式给出精确导数。
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x, A=2.0, B=0.5, C=1.0):
    """f(x) = A sin(x + B) + C"""
    return A * np.sin(x + B) + C


def f_exact_deriv(x, A=2.0, B=0.5):
    """精确导数 f'(x) = A cos(x + B)"""
    return A * np.cos(x + B)


def finite_diff(f, x, delta, tau):
    """一般中心差分"""
    return (f(x + delta) - f(x - delta)) / tau


def parameter_shift(f, x):
    """参数平移法则: δ=π/2, τ=2"""
    return (f(x + np.pi / 2) - f(x - np.pi / 2)) / 2.0


# 测试参数
A_test, B_test, C_test = 2.0, 0.5, 1.0

# ============================================================
# 验证
# ============================================================
print("=" * 60)
print("参数平移法则验证")
print("=" * 60)

x_vals = np.linspace(0, 2 * np.pi, 13)
for x in x_vals:
    exact = f_exact_deriv(x, A_test, B_test)
    ps = parameter_shift(lambda t: f(t, A_test, B_test, C_test), x)
    fd = finite_diff(lambda t: f(t, A_test, B_test, C_test), x, delta=1e-6, tau=2e-6)
    print(f"  x={x/np.pi:.2f}π: 精确={exact:.6f}, 参数平移={ps:.6f}, "
          f"误差={abs(ps-exact):.2e}, 有限差分误差={abs(fd-exact):.2e}")

# ============================================================
# 可视化
# ============================================================
x_plot = np.linspace(-np.pi, 3 * np.pi, 500)
f_vals = f(x_plot, A_test, B_test, C_test)
exact_deriv = f_exact_deriv(x_plot, A_test, B_test)
ps_deriv = [parameter_shift(lambda t: f(t, A_test, B_test, C_test), x) for x in x_plot]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.plot(x_plot / np.pi, f_vals, label=r"$f(x) = A\sin(x+B)+C$", lw=2)
ax1.set_ylabel("f(x)")
ax1.set_title("原函数")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x_plot / np.pi, exact_deriv, label=r"解析导数 $f'(x) = A\cos(x+B)$", lw=2)
ax2.plot(x_plot / np.pi, ps_deriv, '--', label=r"参数平移: $\delta=\pi/2, \tau=2$", lw=2)
ax2.set_xlabel(r"$x/\pi$")
ax2.set_ylabel("f'(x)")
ax2.set_title("导数比较")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("3-2_参数平移.png", dpi=150, bbox_inches="tight")
plt.show()

# 结论
print("\n" + "=" * 60)
print("结论：")
print("=" * 60)
print("""
对于 f(x) = A sin(x+B) + C:
  取 δ = π/2, τ = 2 可使中心差分精确等于解析导数。

原因：
  sin(x+δ+B) - sin(x-δ+B) = 2 sin(δ) cos(x+B)
  当 sin(δ)/τ = 1/2 时精确成立。
  取 δ = π/2 → τ = 2 为一组特解。
  (更一般地，任意满足 τ = 2 sin(δ) [δ ≠ 0] 都行)

这是量子计算中"参数平移法则"(parameter shift rule)的数学基础。
""")
