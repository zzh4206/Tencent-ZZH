"""
星火预习题目 — OPT-1-2：Grover 算法实现
===========================================
实现 Grover 搜索线路，对 3×2 格子染色问题搜索所有解。

Grover 算法步骤：
1. 准备均匀叠加态 H^{⊗n}|0⟩
2. 应用 Grover 迭代 G = H^{⊗n} S₀ H^{⊗n} · O
   其中 O 是相位 oracle，S₀ 是绕 |0⟩ 的反射
3. 迭代 k ≈ (π/4)√(N/M) 次，N=2ⁿ, M=解的数量
4. 测量

对 n=6, M=2:
  N = 2⁶ = 64
  k_opt = ⌊(π/4)√(64/2)⌋ = ⌊(π/4)×√32⌋ = ⌊(π/4)×5.657⌋ ≈ ⌊4.44⌋ = 4
"""

import numpy as np
import tensorcircuit as tc
import matplotlib.pyplot as plt

K = tc.set_backend("tensorflow")

# 问题参数
n_qubits = 6           # 数据 qubit 数
N = 2 ** n_qubits      # 搜索空间大小
M = 2                  # 正确解的数量

# 两个解
SOLUTION_A = 21  # 010101
SOLUTION_B = 42  # 101010
SOLUTIONS = [SOLUTION_A, SOLUTION_B]

# 最优迭代次数
k_opt = int(np.round((np.pi / 4) * np.sqrt(N / M)))
print(f"搜索空间: N = 2^{n_qubits} = {N}")
print(f"目标解数: M = {M}")
print(f"最优迭代次数: k_opt ≈ {k_opt}")

# ============================================================
# 构建 Grover 线路的各组件
# ============================================================


def prepare_superposition(c):
    """对所有 6 个 qubit 施加 H 门"""
    for i in range(n_qubits):
        c.h(i)


def phase_oracle(c, target_ints):
    """
    相位 oracle：对目标态翻转相位。
    使用 X 门 + 多控 Z 门。
    """
    for target in target_ints:
        bits = [(target >> i) & 1 for i in range(n_qubits)]
        # X 门翻转 0 位，使目标态变成 |11...1⟩
        for i in range(n_qubits):
            if bits[i] == 0:
                c.x(i)
        # 多控 Z（等效于对 |11...1⟩ 加相位 -1）
        # 用 controlled-Z 实现：先 H, 再多控 X, 再 H
        c.h(n_qubits - 1)
        # multicontrol X gate on first n_qubits-1, target last
        ctrl_qubits = list(range(n_qubits - 1))
        c.multicontrol(*ctrl_qubits, n_qubits - 1,
                       ctrl=ctrl_qubits,
                       unitary=tc.gates._x)
        c.h(n_qubits - 1)
        # 复原 X 门
        for i in range(n_qubits):
            if bits[i] == 0:
                c.x(i)


def diffusion_operator(c):
    """
    绕均匀叠加态的反射：H^{⊗n} (2|0⟩⟨0| - I) H^{⊗n}
    等效于：H^{⊗n} 后在除某辅助位外的所有 qubit 上施加多控 Z
    简化：H ⊗ X ⊗ H = Z 前加 X 的模式
    """
    # 绕 |0⟩ 反射: 2|0⟩⟨0| - I
    # 用 X 门 + 多控 Z 实现
    for i in range(n_qubits):
        c.h(i)
    for i in range(n_qubits):
        c.x(i)
    # 多控 Z
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
    """构建 k 次 Grover 迭代的线路"""
    c = tc.Circuit(n_qubits)
    prepare_superposition(c)
    for _ in range(k):
        phase_oracle(c, SOLUTIONS)
        diffusion_operator(c)
    return c


# ============================================================
# 模拟：不同迭代次数下的成功概率
# ============================================================
k_values = list(range(11))
success_probs = []
all_probs = []

for k in k_values:
    c = grover_circuit(k)
    sv = c.state().numpy().flatten()
    probs = np.abs(sv) ** 2

    # 成功概率 = 测量得到合法解的概率
    p_success = sum(probs[s] for s in SOLUTIONS)
    success_probs.append(p_success)
    all_probs.append(probs)

    # 所有解的概率
    probs_per_sol = {s: probs[s] for s in SOLUTIONS}
    print(f"k={k:2d}: P(success) = {p_success:.4f}  "
          f"(sol {SOLUTION_A}: {probs_per_sol[SOLUTION_A]:.4f}, "
          f"sol {SOLUTION_B}: {probs_per_sol[SOLUTION_B]:.4f})")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: success probability vs k
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

# Plot 2: probability distribution for key k values
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
# mark solutions
for s in SOLUTIONS:
    ax2.axvline(x=s, color='red', linestyle='--', alpha=0.4, lw=0.8)

# Plot 3: amplitude visualization at optimal k
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

# ============================================================
# 理论对照
# ============================================================
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
