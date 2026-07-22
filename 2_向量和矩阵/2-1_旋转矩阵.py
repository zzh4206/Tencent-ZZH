"""
星火预习题目 — 2-1：旋转矩阵
==============================
给定 v = (1,0), R(θ) = [[cosθ, sinθ], [-sinθ, cosθ]],
计算 v' = R(θ)v，可视化其几何意义。

结论：R(θ) 是顺时针旋转矩阵，将向量 (1,0) 顺时针旋转 θ 角。
"""

import numpy as np
import matplotlib.pyplot as plt


def R(theta):
    """旋转矩阵 R(θ) = [[cosθ, sinθ], [-sinθ, cosθ]]"""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, s], [-s, c]])


v = np.array([1.0, 0.0])

# 展示几个不同角度的旋转
thetas = [0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
ax.axhline(0, color="gray", lw=0.5)
ax.axvline(0, color="gray", lw=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(r"$v' = R(\theta)v$ — 顺时针旋转")

colors = plt.cm.viridis(np.linspace(0, 1, len(thetas)))
for theta, c in zip(thetas, colors):
    vp = R(theta) @ v
    ax.arrow(0, 0, vp[0], vp[1], head_width=0.05, head_length=0.05,
             fc=c, ec=c, alpha=0.8, label=rf"$\theta={theta:.2f}$")

ax.legend(fontsize=8)
plt.savefig("2-1_旋转矩阵.png", dpi=150, bbox_inches="tight")
plt.show()

# 连续扫描 θ
thetas_cont = np.linspace(0, 2 * np.pi, 200)
x_vals = np.cos(thetas_cont)   # R(θ)v 的第一分量
y_vals = -np.sin(thetas_cont)  # R(θ)v 的第二分量

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(thetas_cont, x_vals, label="x'")
ax1.plot(thetas_cont, y_vals, label="y'")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel("分量值")
ax1.legend()
ax1.set_title("旋转后分量随 θ 变化")

ax2.plot(x_vals, y_vals)
ax2.set_aspect("equal")
ax2.set_xlabel("x'")
ax2.set_ylabel("y'")
ax2.set_title("旋转后向量轨迹 (单位圆，顺时针)")
plt.tight_layout()
plt.savefig("2-1_旋转矩阵_连续.png", dpi=150, bbox_inches="tight")
plt.show()

print("结论：R(θ) = [[cosθ, sinθ], [-sinθ, cosθ]] 是顺时针旋转矩阵。")
print("对于 v=(1,0)，v' = (cosθ, -sinθ)，即把 x 轴单位向量顺时针转 θ。")
