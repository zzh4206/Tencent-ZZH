import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


def find_min_distance_point(points, learning_rate=0.01, n_steps=500):
    points_t = K.convert_to_tensor(points.astype(np.float32))

    def total_distance(xy):
        x, y = xy[0], xy[1]
        dx = x - points_t[:, 0]
        dy = y - points_t[:, 1]
        dists = K.sqrt(dx ** 2 + dy ** 2)
        return K.sum(dists)

    grad_fn = K.grad(total_distance)


    xy = K.convert_to_tensor(points.mean(axis=0).astype(np.float32))
    history_xy = [xy.numpy().copy()]
    history_loss = [float(total_distance(xy).numpy())]

    for step in range(n_steps):
        g = grad_fn(xy)
        lr = learning_rate
        xy_new = xy - lr * g
        xy = xy_new

        history_xy.append(xy.numpy().copy())
        loss = float(total_distance(xy).numpy())
        history_loss.append(loss)

        if K.max(K.abs(g)) < 1e-7:
            break

    return xy.numpy(), np.array(history_xy), history_loss


np.random.seed(42)
n_points = 8
points = np.random.uniform(0, 10, (n_points, 2))

x_opt, y_opt = find_min_distance_point(points)[:2]
x_opt = x_opt[0]
y_opt = x_opt[1] if len(x_opt) > 1 else y_opt


x_opt, hist_xy, hist_loss = find_min_distance_point(points)
x_opt_val = float(x_opt[0]) if hasattr(x_opt, '__len__') else float(x_opt)
y_opt_val = float(x_opt[1]) if len(x_opt) > 1 else float(y_opt)

print(f"输入点 ({n_points} 个):")
for i, (xi, yi) in enumerate(points):
    print(f"  P{i}: ({xi:.3f}, {yi:.3f})")


opt_xy, hist_xy, hist_loss = find_min_distance_point(points)
print(f"\n最优井位: ({opt_xy[0]:.4f}, {opt_xy[1]:.4f})")
total_d = np.sum(np.sqrt((points[:, 0] - opt_xy[0])**2 + (points[:, 1] - opt_xy[1])**2))
print(f"总距离: {total_d:.4f}")


centroid = points.mean(axis=0)
centroid_d = np.sum(np.sqrt((points[:, 0] - centroid[0])**2 + (points[:, 1] - centroid[1])**2))
print(f"质心: ({centroid[0]:.4f}, {centroid[1]:.4f}), 总距离: {centroid_d:.4f}")
print(f"相对改进: {(1 - total_d/centroid_d)*100:.1f}%")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))


ax1.scatter(points[:, 0], points[:, 1], c='blue', s=80, zorder=5,
            edgecolors='black', label=f'{n_points} villages')
ax1.scatter([opt_xy[0]], [opt_xy[1]], c='red', s=200, marker='*',
            zorder=10, edgecolors='black', linewidths=2,
            label=f'Optimal well ({opt_xy[0]:.2f}, {opt_xy[1]:.2f})')
ax1.scatter([centroid[0]], [centroid[1]], c='green', s=80, marker='s',
            zorder=5, edgecolors='black',
            label=f'Centroid ({centroid[0]:.2f}, {centroid[1]:.2f})')


for i, (xi, yi) in enumerate(points):
    ax1.plot([opt_xy[0], xi], [opt_xy[1], yi], 'r--', alpha=0.3, lw=1)
    ax1.annotate(f'P{i}', (xi, yi), textcoords="offset points",
                 xytext=(5, 5), fontsize=8)


x_grid = np.linspace(0, 10, 100)
y_grid = np.linspace(0, 10, 100)
XX, YY = np.meshgrid(x_grid, y_grid)
ZZ = np.zeros_like(XX)
for i in range(len(x_grid)):
    for j in range(len(y_grid)):
        dx = points[:, 0] - XX[i, j]
        dy = points[:, 1] - YY[i, j]
        ZZ[i, j] = np.sum(np.sqrt(dx**2 + dy**2))

ax1.contour(XX, YY, ZZ, levels=15, colors='gray', alpha=0.3, linewidths=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Geometric Median (dig a well)')
ax1.legend(fontsize=8)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)


ax2.plot(hist_loss, 'b-', lw=1.5)
ax2.axhline(y=total_d, color='red', linestyle='--', alpha=0.5,
            label=f'Final: {total_d:.3f}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Total distance')
ax2.set_title('Convergence of gradient descent')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.suptitle('Variational Optimization: Geometric Median', fontsize=14)
plt.tight_layout()
plt.savefig("OPT-2-2_优化点.png", dpi=150, bbox_inches="tight")
plt.show()
