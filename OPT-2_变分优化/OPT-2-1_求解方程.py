import numpy as np
import tensorcircuit as tc

K = tc.set_backend("tensorflow")


def solve_cubic(a, b, c, d, x0=None, learning_rate=0.01, n_steps=500, tol=1e-8):
    a_t = K.convert_to_tensor(a, dtype=K.float32)
    b_t = K.convert_to_tensor(b, dtype=K.float32)
    c_t = K.convert_to_tensor(c, dtype=K.float32)
    d_t = K.convert_to_tensor(d, dtype=K.float32)

    def loss_fn(x):
        poly = a_t * x ** 3 + b_t * x ** 2 + c_t * x + d_t
        return poly ** 2

    grad_fn = K.grad(loss_fn)

    def find_root_from(x0_val):
        x = K.convert_to_tensor(x0_val, dtype=K.float32)
        history = [x.numpy()]
        for step in range(n_steps):
            g = grad_fn(x)

            lr_eff = learning_rate / (1.0 + 0.01 * step)
            x_new = x - lr_eff * g

            x_new = K.clip(x_new, -100.0, 100.0)
            x = x_new
            history.append(x.numpy())
            if abs(g.numpy()) < tol:
                break
        return x.numpy(), history


    if x0 is not None:
        root, _ = find_root_from(x0)
        return [root]
    else:
        candidates = [-10, -5, -2, 0, 2, 5, 10]
        roots_raw = []
        for x0_cand in candidates:
            r, _ = find_root_from(x0_cand)
            roots_raw.append(r)


        roots = []
        for r in roots_raw:
            is_new = True
            for existing in roots:
                if abs(r - existing) < 1e-4:
                    is_new = False
                    break
            if is_new:
                roots.append(r)
        return sorted(roots)


print("=" * 60)
print("变分优化求解三次方程")
print("=" * 60)

test_cases = [

    (1, -6, 11, -6, "x³-6x²+11x-6=0, 根: 1,2,3"),
    (1, 0, -1, 0, "x³-x=0, 根: -1,0,1"),
    (1, 1, 1, 1, "x³+x²+x+1=0, 复根为主, 实根: -1"),
    (1, -3, 3, -1, "(x-1)³=0, 三重根: 1"),
    (2, -4, -6, 8, "2x³-4x²-6x+8=0"),
]

for a, b, c, d, desc in test_cases:
    roots = solve_cubic(a, b, c, d)

    residuals = [abs(a * r ** 3 + b * r ** 2 + c * r + d) for r in roots]
    max_res = max(residuals) if residuals else 0
    print(f"\n{desc}")
    print(f"  找到的实根: {[f'{r:.6f}' for r in roots]}")
    print(f"  最大残差: {max_res:.2e}")


    np_roots = np.roots([a, b, c, d])
    np_real = sorted([r.real for r in np_roots if abs(r.imag) < 1e-8])
    print(f"  numpy 实根: {[f'{r:.6f}' for r in np_real]}")


import matplotlib.pyplot as plt

a, b, c, d = 1, -6, 11, -6

a_t = K.convert_to_tensor(a, dtype=K.float32)
b_t = K.convert_to_tensor(b, dtype=K.float32)
c_t = K.convert_to_tensor(c, dtype=K.float32)
d_t = K.convert_to_tensor(d, dtype=K.float32)


def loss_fn_v(x):
    poly = a_t * x ** 3 + b_t * x ** 2 + c_t * x + d_t
    return poly ** 2


grad_fn_v = K.grad(loss_fn_v)

x_plot = np.linspace(-0.5, 4.0, 500)
loss_plot = []
for xv in x_plot:
    x_t = K.convert_to_tensor(xv, dtype=K.float32)
    loss_plot.append(float(loss_fn_v(x_t).numpy()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))


ax1.semilogy(x_plot, loss_plot, 'b-', lw=1, alpha=0.3, label='Loss landscape')
ax1.set_ylabel('Loss L(x)')
ax1.set_xlabel('x')
ax1.set_title('Gradient descent on cubic equation')
ax1.grid(True, alpha=0.3)


for r in [1, 2, 3]:
    ax1.axvline(x=r, color='green', linestyle='--', alpha=0.4)
    ax1.annotate(f'x={r}', xy=(r, 1e-10), fontsize=9, color='green',
                 ha='center')


init_points = [-0.2, 1.5, 2.5, 3.5, 4.0]
colors = plt.cm.tab10(np.linspace(0, 1, len(init_points)))
for x0, color in zip(init_points, colors):
    x = K.convert_to_tensor(x0, dtype=K.float32)
    traj_x = [x.numpy()]
    traj_l = [float(loss_fn_v(x).numpy())]
    for step in range(200):
        g = grad_fn_v(x)
        lr = 0.01
        x_new = x - lr * g
        x_new = K.clip(x_new, -10.0, 10.0)
        x = x_new
        traj_x.append(x.numpy())
        traj_l.append(float(loss_fn_v(x).numpy()))
        if abs(g.numpy()) < 1e-7:
            break
    ax1.semilogy(traj_x, traj_l, '.-', color=color, markersize=2, lw=0.8,
                 label=f'x₀={x0}')
    ax2.plot(range(len(traj_x)), traj_l, '.-', color=color, markersize=2, lw=0.8,
             label=f'x₀={x0}')

ax1.legend(fontsize=7)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss L(x)')
ax2.set_yscale('log')
ax2.set_title('Convergence curves')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=7)

plt.suptitle('Variational approach: solving ax³+bx²+cx+d=0', fontsize=14)
plt.tight_layout()
plt.savefig("OPT-2-1_求解方程.png", dpi=150, bbox_inches="tight")
plt.show()
