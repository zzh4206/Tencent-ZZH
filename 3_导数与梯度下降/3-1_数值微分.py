import numpy as np


def numerical_gradient(f, x, delta=1e-6, method="central"):
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


def test_f1(x):
    return x[0] ** 2 + 3 * x[1] ** 2 + 2 * x[0] * x[1]


def analytical_grad_f1(x):
    return np.array([
        2 * x[0] + 2 * x[1],
        6 * x[1] + 2 * x[0]
    ])


def test_f2(x):
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
