"""
星火预习题目 — 3-1：数值微分
===============================
实现一个函数，用数值差分 (f(x+δ) - f(x)) / δ 计算
任意函数 f 关于输入向量各分量的导数。

提供三种差分方案：
1. 前向差分： (f(x+δe_i) - f(x)) / δ
2. 后向差分： (f(x) - f(x-δe_i)) / δ
3. 中心差分： (f(x+δe_i) - f(x-δe_i)) / (2δ)  [更精确，O(δ²)]
"""

import numpy as np


def numerical_gradient(f, x, delta=1e-6, method="central"):
    """
    计算 f(x) 关于向量 x 各分量的数值导数。

    Parameters
    ----------
    f : callable
        输入向量 x，输出标量
    x : np.ndarray
        输入向量
    delta : float
        差分步长
    method : str
        "forward" / "backward" / "central"

    Returns
    -------
    grad : np.ndarray
        各分量的偏导数，形状与 x 相同
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for i in range(len(x)):
        ei = np.zeros_like(x)
        ei[i] = 1.0

        if method == "forward":
            grad[i] = (f(x + delta * ei) - f(x)) / delta
        elif method == "backward":
            grad[i] = (f(x) - f(x - delta * ei)) / delta
        elif method == "central":
            grad[i] = (f(x + delta * ei) - f(x - delta * ei)) / (2 * delta)
        else:
            raise ValueError(f"Unknown method: {method}")

    return grad


# ============================================================
# 测试
# ============================================================
def test_f1(x):
    """f = x₀² + 3x₁² + 2x₀x₁"""
    return x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[0] * x[1]


def analytical_grad_f1(x):
    """解析梯度"""
    return np.array([
        2 * x[0] + 2 * x[1],
        6 * x[1] + 2 * x[0]
    ])


def test_f2(x):
    """f = sin(x₀)cos(x₁)"""
    return np.sin(x[0]) * np.cos(x[1])


def analytical_grad_f2(x):
    return np.array([
        np.cos(x[0]) * np.cos(x[1]),
        -np.sin(x[0]) * np.sin(x[1])
    ])


print("=" * 60)
print("测试数值微分")
print("=" * 60)

test_points = [
    np.array([1.0, 2.0]),
    np.array([0.5, -1.5]),
    np.array([np.pi / 3, np.pi / 4]),
]

for f, analytical, name in [(test_f1, analytical_grad_f1, "x₀²+3x₁²+2x₀x₁"),
                              (test_f2, analytical_grad_f2, "sin(x₀)cos(x₁)")]:
    print(f"\n--- {name} ---")
    for x_test in test_points:
        for method in ["forward", "central"]:
            num_grad = numerical_gradient(f, x_test, method=method)
            ana_grad = analytical(x_test)
            err = np.max(np.abs(num_grad - ana_grad))
            print(f"  x={x_test}, method={method:8s}, max err={err:.2e}")
            print(f"    num: {num_grad}, ana: {ana_grad}")

# 步长对精度的影响
print("\n" + "=" * 60)
print("步长 δ 对中心差分精度的影响")
print("=" * 60)
x_test = np.array([1.0, 2.0])
ana = analytical_grad_f1(x_test)
for delta in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    num = numerical_gradient(test_f1, x_test, delta=delta, method="central")
    err = np.max(np.abs(num - ana))
    print(f"  δ={delta:.0e}: error={err:.2e}")

print("\n最佳 δ 通常在 1e-6 到 1e-8 之间，")
print("太小则浮点舍入误差主导，太大则截断误差主导。")
