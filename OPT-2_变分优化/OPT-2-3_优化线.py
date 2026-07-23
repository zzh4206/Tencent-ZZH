import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


def find_best_line(points, learning_rate=0.001, n_steps=1000, epsilon=1e-4):
    points_t = K.convert_to_tensor(points.astype(np.float32))

    def total_distance(params):
        a, b, c = params[0], params[1], params[2]
        z = a * points_t[:, 0] + b * points_t[:, 1] + c
        smooth_abs = K.sqrt(z ** 2 + epsilon)

        norm_penalty = 10.0 * (a ** 2 + b ** 2 - 1.0) ** 2
        return K.sum(smooth_abs) + norm_penalty

    grad_fn = K.grad(total_distance)


    x_mean = points[:, 0].mean()
    y_mean = points[:, 1].mean()

    centered = points - np.array([x_mean, y_mean])
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[1]
    a0, b0 = normal[0], normal[1]

    norm = np.sqrt(a0**2 + b0**2)
    a0, b0 = a0 / norm, b0 / norm
    c0 = -(a0 * x_mean + b0 * y_mean)

    params = K.convert_to_tensor(np.array([a0, b0, c0], dtype=np.float32))
    history = [params.numpy().copy()]
    history_loss = [float(total_distance(params).numpy())]

    for step in range(n_steps):
        g = grad_fn(params)
        lr = learning_rate * (1.0 / (1.0 + 0.001 * step))
        params_new = params - lr * g

        a, b, c = params_new[0], params_new[1], params_new[2]
        norm_ab = K.sqrt(a ** 2 + b ** 2)
        a = a / norm_ab
        b = b / norm_ab
        params = K.stack([a, b, c])

        history.append(params.numpy().copy())
        loss = float(total_distance(params).numpy())
        history_loss.append(loss)

        if K.max(K.abs(g)) < 1e-8:
            break

    a_opt, b_opt, c_opt = params.numpy()
    return a_opt, b_opt, c_opt, np.array(history), history_loss


np.random.seed(42)
n_points = 10

x_raw = np.random.uniform(0, 10, n_points)
true_slope = 1.5
true_intercept = 2.0
y_raw = true_slope * x_raw + true_intercept + np.random.normal(0, 1.5, n_points)
points = np.column_stack([x_raw, y_raw])

print("=" * 60)
print("变分优化：找最优直线（多村庄通公路）")
print("=" * 60)
print(f"输入 {n_points} 个点:")

a_opt, b_opt, c_opt, history, hist_loss = find_best_line(points)

print(f"\n最优直线: {a_opt:.4f}x + {b_opt:.4f}y + {c_opt:.4f} = 0")

if abs(b_opt) > 1e-6:
    m_opt = -a_opt / b_opt
    k_opt = -c_opt / b_opt
    print(f"  y = {m_opt:.4f}x + {k_opt:.4f}")
else:
    print(f"  x = {-c_opt/a_opt:.4f} (vertical)")

print(f"真实直线: y = {true_slope}x + {true_intercept}")


distances = np.abs(a_opt * points[:, 0] + b_opt * points[:, 1] + c_opt)
total_d = np.sum(distances)
print(f"总距离 (L1): {total_d:.4f}")


A = np.column_stack([x_raw, np.ones_like(x_raw)])
m_ls, k_ls = np.linalg.lstsq(A, y_raw, rcond=None)[0]
ls_distances = np.abs(m_ls * x_raw - y_raw + k_ls) / np.sqrt(m_ls**2 + 1)
ls_total = np.sum(ls_distances)
print(f"\n最小二乘线: y = {m_ls:.4f}x + {k_ls:.4f}")
print(f"最小二乘总距离 (L1): {ls_total:.4f}")
print(f"变分方法改进: {(1 - total_d/ls_total)*100:.1f}%")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))


x_plot = np.linspace(min(x_raw) - 1, max(x_raw) + 1, 200)

ax1.scatter(points[:, 0], points[:, 1], c='blue', s=80, zorder=5,
            edgecolors='black', label=f'{n_points} villages')


if abs(b_opt) > 1e-6:
    y_opt_line = m_opt * x_plot + k_opt
    ax1.plot(x_plot, y_opt_line, 'r-', lw=2,
             label=f'Variational: y={m_opt:.2f}x+{k_opt:.2f}')
else:
    x_v = -c_opt / a_opt
    ax1.axvline(x=x_v, color='r', lw=2, label=f'Variational: x={x_v:.2f}')


y_ls_line = m_ls * x_plot + k_ls
ax1.plot(x_plot, y_ls_line, 'g--', lw=1.5,
         label=f'Least squares: y={m_ls:.2f}x+{k_ls:.2f}')


y_true_line = true_slope * x_plot + true_intercept
ax1.plot(x_plot, y_true_line, 'k:', lw=1, alpha=0.5,
         label=f'True: y={true_slope}x+{true_intercept}')


for i, (xi, yi) in enumerate(points):

    d = a_opt * xi + b_opt * yi + c_opt
    x_proj = xi - a_opt * d
    y_proj = yi - b_opt * d
    ax1.plot([xi, x_proj], [yi, y_proj], 'r-', alpha=0.2, lw=0.8)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Best-fit line (minimize L1 distance)')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)


ax2.plot(hist_loss, 'b-', lw=1.5)
ax2.axhline(y=total_d, color='red', linestyle='--', alpha=0.5,
            label=f'Final: {total_d:.3f}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total distance')
ax2.set_title('Convergence of gradient descent')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.suptitle('Variational Optimization: Best Line (minimize L1 distance)',
             fontsize=14)
plt.tight_layout()
plt.savefig("OPT-2-3_优化线.png", dpi=150, bbox_inches="tight")
plt.show()
