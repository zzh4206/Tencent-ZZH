import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")


n_qubits = 6
N = 2 ** n_qubits
M = 2


SOLUTION_A = 21
SOLUTION_B = 42
SOLUTIONS = [SOLUTION_A, SOLUTION_B]


k_opt = int(np.round((np.pi / 4) * np.sqrt(N / M)))
print(f"搜索空间: N = 2^{n_qubits} = {N}")
print(f"目标解数: M = {M}")
print(f"最优迭代次数: k_opt ≈ {k_opt}")


def prepare_superposition(c):
    for i in range(n_qubits):
        c.h(i)


def phase_oracle(c, target_ints):
    for target in target_ints:
        bits = [(target >> i) & 1 for i in range(n_qubits)]

        for i in range(n_qubits):
            if bits[i] == 0:
                c.x(i)


        c.h(n_qubits - 1)

        ctrl_qubits = list(range(n_qubits - 1))
        c.multicontrol(*ctrl_qubits, n_qubits - 1,
                       ctrl=ctrl_qubits,
                       unitary=tc.gates._x)
        c.h(n_qubits - 1)

        for i in range(n_qubits):
            if bits[i] == 0:
                c.x(i)


def diffusion_operator(c):


    for i in range(n_qubits):
        c.h(i)
    for i in range(n_qubits):
        c.x(i)

    c.h(n_qubits - 1)
    ctrl_qubits = list(range(n_qubits - 1))
    c.multicontrol(*ctrl_qubits, n_qubits - 1,
                   ctrl=ctrl_qubits,
                   unitary=tc.gates._x)
    c.h(n_qubits - 1)
    for i in range(n_qubits):
        c.x(i)
    for i in range(n_qubits):
        c.h(i)


def grover_circuit(k):
    c = tc.Circuit(n_qubits)
    prepare_superposition(c)
    for _ in range(k):
        phase_oracle(c, SOLUTIONS)
        diffusion_operator(c)
    return c


k_values = list(range(11))
success_probs = []
all_probs = []

for k in k_values:
    c = grover_circuit(k)
    sv = c.state().numpy().flatten()
    probs = np.abs(sv) ** 2


    p_success = sum(probs[s] for s in SOLUTIONS)
    success_probs.append(p_success)
    all_probs.append(probs)


    probs_per_sol = {s: probs[s] for s in SOLUTIONS}
    print(f"k={k:2d}: P(success) = {p_success:.4f}  "
          f"(sol {SOLUTION_A}: {probs_per_sol[SOLUTION_A]:.4f}, "
          f"sol {SOLUTION_B}: {probs_per_sol[SOLUTION_B]:.4f})")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))


ax1 = axes[0]
ax1.plot(k_values, success_probs, 'o-', lw=2, markersize=8, color='C0')
ax1.axvline(x=k_opt, color='red', linestyle='--',
            label=f'$k_{{opt}}={k_opt}$')
ax1.set_xlabel('Number of Grover iterations k')
ax1.set_ylabel('Success probability')
ax1.set_title('Success probability vs iterations')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1.05)


ax2 = axes[1]
key_ks = [0, 1, 2, k_opt, 6]
colors = plt.cm.viridis(np.linspace(0, 1, len(key_ks)))
bar_width = 0.15
x_pos = np.arange(N)
for idx, k in enumerate(key_ks):
    offset = (idx - len(key_ks) / 2 + 0.5) * bar_width
    ax2.bar(x_pos + offset, all_probs[k], width=bar_width,
            color=colors[idx], alpha=0.7, label=f'k={k}')
ax2.set_xlabel('State index')
ax2.set_ylabel('Probability')
ax2.set_title('Probability distribution over all states')
ax2.legend(fontsize=7)

for s in SOLUTIONS:
    ax2.axvline(x=s, color='red', linestyle='--', alpha=0.4, lw=0.8)


ax3 = axes[2]
probs_opt = all_probs[k_opt]
x_idx = np.arange(N)
ax3.stem(x_idx, probs_opt, basefmt='k-', linefmt='C0-', markerfmt='C0o')
ax3.set_xlabel('State index')
ax3.set_ylabel('Probability')
ax3.set_title(f'Final distribution (k={k_opt})')
for s in SOLUTIONS:
    ax3.annotate(f'{probs_opt[s]*100:.1f}%', xy=(s, probs_opt[s]),
                 xytext=(s + 3, probs_opt[s] + 0.05),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red')

plt.suptitle('Grover Search: 3×2 Grid Coloring Problem (n=6, M=2)',
             fontsize=14)
plt.tight_layout()
plt.savefig("OPT-1-2_Grover搜索.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n" + "=" * 60)
print("理论分析")
print("=" * 60)
print(f"""
Grover 算法分析：

1. 搜索空间: N = 2⁶ = 64
   目标解: M = 2

2. 最优迭代次数:
   k_opt ≈ (π/4)√(N/M) = (π/4)√(64/2)
         ≈ (π/4) × 5.657
         ≈ 4.44 → k_opt = 4

3. 成功概率随 k 振荡:
   第一次峰值在 k_opt = 4，概率应接近 1。
   周期 ≈ (π/2)√(N/M) ≈ 8.89

4. 经典 vs 量子:
   经典随机猜: P = M/N = 2/64 = 3.125%，需 ~32 次查询
   Grover: ~4 次迭代达到 ~100%，平方加速 √(N/M) → 4 vs 32

5. 实际 k 值小于理论最优的可能原因（取决于多控门的具体实现方式）。
""")
